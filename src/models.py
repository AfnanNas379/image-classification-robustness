"""The three candidate models, all for 32x32 RGB input and 10 classes.

Owner: Lulu (models).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- plain CNN (baseline)
def _conv_bn_relu(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class PlainCNN(nn.Module):
    """Six 3x3 conv layers (64-64-128-128-256-256), BN, ReLU, max-pool, GAP, dropout. ~1.15 M params."""

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_bn_relu(3, 64), *_conv_bn_relu(64, 64), nn.MaxPool2d(2),        # 32 -> 16
            *_conv_bn_relu(64, 128), *_conv_bn_relu(128, 128), nn.MaxPool2d(2),    # 16 -> 8
            *_conv_bn_relu(128, 256), *_conv_bn_relu(256, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------- ResNet-18 (CIFAR variant)
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))           # residual (skip) connection


class ResNet18(nn.Module):
    """He et al. (2016) ResNet-18 with a CIFAR stem (3x3 conv, no max-pool). ~11.2 M params."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        layers, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------------------------------------------------------- Vision Transformer
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """Dosovitskiy et al. (2021) ViT: 4x4 patches, D=384, 7 blocks, 6 heads, MLP 1536. ~12.5 M params.

    The standard patch embedding is used (no PreLayerNorm), as stated in the report.
    """

    def __init__(self, num_classes=10, image_size=32, patch_size=4, dim=384, depth=7,
                 heads=6, mlp_dim=1536, dropout=0.1):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2                       # 64 tokens
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)      # linear patch projection
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim)) # learnable positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)               # (B, 64, D)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        x = self.blocks(self.drop(x))
        return self.head(self.norm(x)[:, 0])                              # classify from [CLS]


def build_model(name: str, cfg: dict | None = None) -> nn.Module:
    cfg = cfg or {}
    if name == "plain_cnn":
        return PlainCNN(dropout=cfg.get("dropout", 0.3))
    if name == "resnet18":
        return ResNet18()
    if name == "vit":
        return ViT(**{k: cfg[k] for k in ("patch_size", "dim", "depth", "heads", "mlp_dim", "dropout") if k in cfg})
    raise ValueError(f"unknown model {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_NAMES = ("plain_cnn", "resnet18", "vit")
