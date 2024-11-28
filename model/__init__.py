from .dgtr import DGTR
def build_model(cfg):
    if cfg.name == "dgtr":
        return DGTR(cfg)
    else:
        raise Exception("NOT VAILD MOODEL NAME")
