"""Backbone (any timm model) + Dropout -> Linear head."""
import timm
import torch
import torch.nn as nn

from config import CFG


class FlagNet(nn.Module):
    def __init__(self, num_classes, backbone=None, pretrained=None, dropout=None):
        super().__init__()
        t = CFG.train
        backbone = backbone or t.backbone
        kwargs = {"img_size": CFG.aug.out_size} if "dinov2" in backbone else {}
        self.backbone = timm.create_model(
            backbone, pretrained=t.pretrained if pretrained is None else pretrained, num_classes=0, **kwargs
        )
        # measure the pooled feature size (timm's num_features is not always what forward() returns,
        # e.g. mobilenetv3 adds a 1280-d conv_head after the 960-d stem features)
        with torch.no_grad():
            self.backbone.eval()
            feat_dim = self.backbone(torch.zeros(1, 3, CFG.aug.out_size, CFG.aug.out_size)).shape[1]
        self.head = nn.Sequential(
            nn.Dropout(t.dropout if dropout is None else dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.backbone_name = backbone

    def forward(self, x):
        return self.head(self.backbone(x))


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FlagNet(ckpt["num_classes"], backbone=ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
