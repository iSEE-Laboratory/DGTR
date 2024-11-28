from .pointnet2 import Pointnet2Backbone


def build_backbone(backbone_cfg):
    if backbone_cfg.name.lower() == "pointnet2":
        return Pointnet2Backbone(backbone_cfg)
    else:
        raise NotImplementedError(f"No such backbone: {backbone_cfg.name}")
