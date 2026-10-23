"""
Modelos 2.5D para clasificación multietiqueta de volúmenes.

Esquema común:

    volumen (B, S, 3, H, W) -> codificador 2D por corte -> GAP -> (B, S, C)
    -> agregación de cortes (atención con compuerta, máximo o media) -> (B, C)
    -> dropout + capa lineal -> K logits (sigmoide independiente por hallazgo)

Codificadores:
    ScratchResNet       red residual propia, inicializada desde cero (modelo A)
    TorchvisionEncoder  ResNet-18/34/50 o EfficientNet-B0 preentrenados en ImageNet (modelo B)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Codificador propio
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """Bloque residual básico (He et al., 2016) con proyección 1x1 si cambian las dimensiones."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ScratchResNet(nn.Module):
    """
    Codificador residual: stem conv 7x7/2 + max pooling y cuatro etapas de dos bloques
    básicos; cada etapa (salvo la primera) reduce la resolución a la mitad.
    Con ``channels=(32, 64, 128, 256)`` tiene ~2.8 M parámetros.
    """

    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        c0 = channels[0]
        self.stem = nn.Sequential(nn.Conv2d(3, c0, 7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(c0), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, stride=2, padding=1))
        layers, cin = [], c0
        for i, c in enumerate(channels):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(cin, c, stride), BasicBlock(c, c, 1)]
            cin = c
        self.layers = nn.Sequential(*layers)
        self.out_dim = channels[-1]
        for m in self.modules():  # inicialización de He
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward_features(self, x):
        return self.layers(self.stem(x))


# ---------------------------------------------------------------------------
# Codificadores preentrenados (torchvision)
# ---------------------------------------------------------------------------
class TorchvisionEncoder(nn.Module):
    """Parte convolucional de un modelo de torchvision, sin la capa de clasificación."""

    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        if name.startswith("resnet"):
            weights = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
                       "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
                       "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[name] if pretrained else None
            net = getattr(tvm, name)(weights=weights)
            self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3, net.layer4)
            self.out_dim = net.fc.in_features
        elif name == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.efficientnet_b0(weights=weights)
            self.body = net.features
            self.out_dim = net.classifier[1].in_features
        else:
            raise ValueError(f"Backbone no soportado: {name}")

    def forward_features(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# Agregación de cortes y modelo completo
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    Agregación por atención con compuerta (Ilse et al., 2018).

    a_s = softmax_s( w^T [tanh(V h_s) * sigmoid(U h_s)] ),  z = sum_s a_s h_s.
    Devuelve ``z`` (B, C) y los pesos ``a`` (B, S).
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = torch.softmax(a, dim=1)
        return (a * h).sum(dim=1), a.squeeze(-1)


class KneeNet(nn.Module):
    """Codificador por corte, agregación de cortes y cabeza lineal multietiqueta."""

    def __init__(self, encoder: nn.Module, n_labels: int = 12, pooling: str = "attention", dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        d = encoder.out_dim
        self.attn = AttentionPooling(d) if pooling == "attention" else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_labels)

    def forward(self, x, return_attention: bool = False):
        B, S = x.shape[:2]
        fmap = self.encoder.forward_features(x.flatten(0, 1))     # (B*S, C, h, w)
        feats = fmap.mean(dim=(2, 3)).view(B, S, -1)              # (B, S, C)
        if self.pooling == "attention":
            study, att = self.attn(feats)
        elif self.pooling == "max":
            study, att = feats.max(dim=1).values, None
        else:
            study, att = feats.mean(dim=1), None
        logits = self.head(self.drop(study))                      # (B, K)
        return (logits, att) if return_attention else logits


def build_model(cfg: dict) -> KneeNet:
    """Modelo según la sección ``model`` del config."""
    m = cfg["model"]
    n_labels = len(cfg["labels"]["names"])
    if m["backbone"] == "scratch":
        enc = ScratchResNet(tuple(m.get("scratch_channels", [32, 64, 128, 256])))
    else:
        enc = TorchvisionEncoder(m["backbone"], bool(m.get("pretrained", True)))
    return KneeNet(enc, n_labels, m.get("pooling", "attention"), float(m.get("dropout", 0.3)))


def count_parameters(model: nn.Module) -> int:
    """Número de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
