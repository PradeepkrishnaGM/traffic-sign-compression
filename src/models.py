"""Teacher (ResNet-18) and student (small CNN) definitions."""
import torch
from torch import nn
from torchvision import models

from . import config as C


def make_teacher(pretrained=True):
    """ResNet-18 with ImageNet weights and a new 43-class output layer."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, C.NUM_CLASSES)
    # ResNet-18 shrinks the image 4x before its first block (stride-2 conv1, then maxpool).
    # On 64x64 inputs that leaves only 16x16 feature maps. Removing the maxpool keeps
    # 32x32 maps, so small sign details survive. conv1 and all pretrained weights stay as they are.
    net.maxpool = nn.Identity()
    return net


def _block(in_ch, out_ch):
    """Two 3x3 convs (each followed by BatchNorm + ReLU), then halve the image size."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Student(nn.Module):
    """About 0.29 M parameters: 3 conv blocks (32/64/128 channels), global pooling, 1 linear layer."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling keeps the linear layer tiny
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, C.NUM_CLASSES)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def make_student():
    return Student()


def make_model(name):
    if name == "teacher":
        return make_teacher()
    if name == "student":
        return make_student()
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
