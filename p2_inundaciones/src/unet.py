"""U-Net para segmentación binaria con conexiones de salto configurables."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Dos convoluciones 3x3 con normalización por lotes y ReLU."""

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                  nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    """
    Compuerta de atención aditiva (Oktay et al., 2018).

    Pondera el mapa de la conexión de salto ``x`` con coeficientes en [0, 1] calculados a partir
    de ``x`` y de la señal del decodificador ``g``.
    """

    def __init__(self, c_g: int, c_x: int, c_int: int):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(c_g, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.wx = nn.Sequential(nn.Conv2d(c_x, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.psi = nn.Sequential(nn.Conv2d(c_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(F.relu(self.wg(g) + self.wx(x)))
        return x * alpha


class UNet(nn.Module):
    """
    U-Net (Ronneberger et al., 2015) con ``depth`` submuestreos.

    ``skip``: ``standard`` concatena el mapa del codificador; ``attention`` lo filtra antes con una
    compuerta de atención; ``none`` elimina las conexiones de salto (codificador-decodificador puro).
    La salida es un logit por píxel.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32, depth: int = 4,
                 skip: str = "standard", dropout: float = 0.0):
        super().__init__()
        if skip not in ("standard", "attention", "none"):
            raise ValueError(f"skip desconocido: {skip}")
        self.skip = skip
        ch = [base_channels * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([ConvBlock(in_channels, ch[0])])
        for i in range(1, depth + 1):
            self.enc.append(ConvBlock(ch[i - 1], ch[i], dropout if i == depth else 0.0))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.gates = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2))
            self.dec.append(ConvBlock(ch[i] * (1 if skip == "none" else 2), ch[i]))
            if skip == "attention":
                self.gates.append(AttentionGate(ch[i], ch[i], max(ch[i] // 2, 8)))
        self.head = nn.Conv2d(ch[0], 1, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            feats.append(x)
        x = feats.pop()
        for k, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = feats.pop()
            if self.skip == "standard":
                x = dec(torch.cat([s, x], dim=1))
            elif self.skip == "attention":
                x = dec(torch.cat([self.gates[k](x, s), x], dim=1))
            else:
                x = dec(x)
        return self.head(x).squeeze(1)


def build_model(cfg: dict) -> UNet:
    """Construye la U-Net según la sección ``model`` del config."""
    m = cfg["model"]
    return UNet(int(m["in_channels"]), int(m["base_channels"]), int(m["depth"]), m["skip"], float(m.get("dropout", 0.0)))


def count_parameters(model: nn.Module) -> int:
    """Número de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
