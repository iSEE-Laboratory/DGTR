from .grasp_queryembed_decoder import GraspQDecoder


def build_decoder(decoder_cfg):
    if decoder_cfg.name.lower() == "graspq":
        return GraspQDecoder(decoder_cfg)
    else:
        raise NotImplementedError(f"No such decode: {decoder_cfg.name}")
