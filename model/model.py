import torch
import torch.nn as nn
import timm

class LD_Baseline(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()

        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return self.sigmoid(self.backbone(x))