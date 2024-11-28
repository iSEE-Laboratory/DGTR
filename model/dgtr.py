from copy import deepcopy
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
from torch.functional import Tensor
from utils.rotation_utils import RotNorm
import pytorch3d.transforms as T

from .backbone import build_backbone
from .decoder import build_decoder
from .loss import GraspLoss
from .utils.helpers import GenericMLP

import time

class DGTR(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbone(cfg.encoder)
        self.decoder = build_decoder(cfg.decoder)
        self.mlp_heads = self.build_mlp_heads(cfg.mlp_heads)
        self.criterion = GraspLoss(cfg.criterion, cfg.device)

        self.loss_weights = cfg.criterion.loss_weights
        self.rotation_type = cfg.rotation_type
        self.rotation_norm = getattr(RotNorm, f"norm_{cfg.rotation_type}", "norm_other")
        self.aux_outputs = cfg.aux_outputs
        self.relative = cfg.relative

    def forward(self, input_dict):
        # t = time.time()
        pred_dict = {}
        pointclouds = input_dict["obj_pc"]
        # encoder output shape: (batch, channel, num_points)
        enc_xyz, enc_features = self.backbone(pointclouds)
        decoder_input = {
            "enc_xyz": enc_xyz,
            "enc_feature": enc_features,
            "point_cloud_dims_min": pointclouds.min(1)[0],
            "point_cloud_dims_max": pointclouds.max(1)[0],
            "obj_pc": pointclouds,
            # "convex_hull": input_dict["convex_hull"],
        }
        # decoder output shape: (num_layers, num_queries, batch, channel)
        query_features, init_hand_poses = self.decoder(decoder_input)
        pred_dict = self.get_predictions(query_features, init_hand_poses)
        # print(time.time()-t)
        loss, loss_dict = self.losses(input_dict, pred_dict)
        return loss, loss_dict, pred_dict

    def losses(self, input_dict, pred_dict):
        loss_dict = self.criterion(pred_dict, input_dict)
        loss = 0
        original_loss_dict = {}  # record losses without weight
        for k, v in loss_dict["outputs"].items():
            loss += v * self.loss_weights[k]
            original_loss_dict[k] = v
            assert original_loss_dict[k] is loss_dict["outputs"][k]
        if "aux_outputs" in loss_dict:
            num_intermediate = len(loss_dict["aux_outputs"])
            for i in range(num_intermediate):
                for k, v in loss_dict["aux_outputs"][i].items():
                    loss += v * self.loss_weights[k]
                    original_loss_dict[f"{k}_{i + 1}"] = v
        return loss, original_loss_dict

    def build_mlp_heads(self, mlp_cfg):
        build_mlp = partial(
            GenericMLP,
            input_dim=mlp_cfg.input_dim,
            hidden_dims=[mlp_cfg.input_dim // 2],
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            dropout=mlp_cfg.dropout_prob,
        )
        mlp_heads = {}
        for name, dim in mlp_cfg.heads.items():
            mlp_heads[name] = build_mlp(output_dim=dim)
        return nn.ModuleDict(mlp_heads)

    def get_predictions(self, feature: Tensor, init_hand_poses: Tensor) -> Dict:
        """
        feature: (num_layers, num_queries, batch, channel)
        init_hand_poses: (batch, num_queries, 29)
        """

        def _inverse_sigmoid(x, eps=1e-3):
            x = x.clamp(min=0, max=1)
            x1 = x.clamp(min=eps)
            x2 = (1 - x).clamp(min=eps)
            return torch.log(x1/x2)
        def _normalize(normed, minmax):
            _min = minmax[..., 0]
            _max = minmax[..., 1]
            return (normed - _min) / (_max - _min) 
        def hook(grad):
            modified_grad = grad * 0.2
            return modified_grad

        # 注册hook到张量x上
        # def _normalize(normed, minmax):
        #     return normed
        feature = feature.permute(0, 2, 3, 1)
        if not self.aux_outputs:
            feature = feature[-1:, ...]  # only use the output feature
        num_layers, B, C, N = feature.size()
        feature = feature.reshape(num_layers * B, C, N)

        outputs = [deepcopy({}) for _ in range(num_layers)]
        if "qpos" in self.mlp_heads.keys():
            pred_qpos = self.mlp_heads["qpos"](feature)
            pred_qpos_norm = pred_qpos.sigmoid()
            pred_qpos_norm = pred_qpos_norm.transpose(-1, -2).reshape(num_layers, B, N, -1)
            for l in range(num_layers):
                if self.relative and 0:
                    ref_qpos = init_hand_poses[..., 7:]
                    outputs[l]["qpos_norm"] = (_inverse_sigmoid(pred_qpos_norm[l]) + 
                                            _inverse_sigmoid(_normalize(
                                            ref_qpos, self.criterion.q_minmax))).sigmoid()
                else:
                    outputs[l]["qpos_norm"] = pred_qpos_norm[l]
        if "rotation" in self.mlp_heads.keys():
            pred_rotation = self.mlp_heads["rotation"](feature)
            pred_rotation_norm = self.rotation_norm(pred_rotation)
            pred_rotation_norm = pred_rotation_norm.transpose(-1, -2).reshape(num_layers, B, N, -1)
            for l in range(num_layers):
                if self.relative and 0:
                    ref_rotation = init_hand_poses[..., 3:7]
                    ref_rotation_matrix = T.quaternion_to_matrix(ref_rotation)
                    rotation_matrix = ref_rotation_matrix @ T.quaternion_to_matrix(pred_rotation_norm[l])
                    outputs[l]["rotation"] = T.matrix_to_quaternion(rotation_matrix)
                else:
                    outputs[l]["rotation"] = pred_rotation_norm[l]
        if "translation" in self.mlp_heads.keys():
            pred_translation = self.mlp_heads["translation"](feature)
            pred_translation_norm = pred_translation.sigmoid()
            pred_translation_norm = pred_translation_norm.transpose(-1, -2).reshape(num_layers, B, N, -1)
            # hook_handle = pred_translation_norm.register_hook(hook)
            for l in range(num_layers):
                if self.relative:
                    ref_trans = init_hand_poses[..., :3]
                    outputs[l]["translation_norm"] = (_inverse_sigmoid(pred_translation_norm[l]) + 
                                                    _inverse_sigmoid(_normalize(
                                                    ref_trans, self.criterion.t_minmax))).sigmoid()
                else:
                    outputs[l]["translation_norm"] = pred_translation_norm[l]
        if "classification" in self.mlp_heads.keys():
            pred_logit = self.mlp_heads["classification"](feature)
            pred_logit = pred_logit.transpose(-1, -2).reshape(num_layers, B, N, -1)
            for l in range(num_layers):
                outputs[l]["logit"] = pred_logit[l]
        feature = feature.transpose(-1, -2).reshape(num_layers, B, N, C)
        for l in range(num_layers):
            outputs[l]["features"] = feature[l]
            outputs[l]["rotation_type"] = self.rotation_type

        if not self.aux_outputs:
            return {
                "outputs": outputs[0],
            }
        else:
            aux_outputs = outputs[:-1]
            outputs = outputs[-1]
            return {
                "outputs": outputs,
                "aux_outputs": aux_outputs,
            }
