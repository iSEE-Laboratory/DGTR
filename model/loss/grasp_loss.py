from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DgnSetFull
from model.utils.hand_model import HandModel
from pytorch3d.loss import chamfer_distance
from torch.functional import Tensor
from torch.nn.parameter import Parameter
import pytorch3d

from utils.rotation_utils import Rot2Axisangle
from .matcher import Matcher


class GraspLoss(nn.Module):
    def __init__(self, loss_cfg, device):
        super().__init__()
        self.hand_model = HandModel(loss_cfg.hand_model, device)
        self.matcher = Matcher(weight_dict=loss_cfg.cost_weights)
        self.loss_weights = {k: v for k, v in loss_cfg.loss_weights.items() if v > 0}

        # norm params
        self.q_minmax = Parameter(DgnSetFull.q_minmax, requires_grad=False)
        self.r_minmax = Parameter(DgnSetFull.r_minmax, requires_grad=False)
        self.t_minmax = Parameter(DgnSetFull.t_minmax, requires_grad=False)
        self.assignments_save = {}

    def assignment_saving(self, assignments, targets):
        for i in range(len(targets["obj_code"])):
            obj_code = targets["obj_code"][i]
            scale = targets["scale"][i]
            per_query_gt_inds = assignments["per_query_gt_inds"][i]
            query_matched_mask = assignments["query_matched_mask"][i]
            if obj_code in self.assignments_save:
                self.assignments_save[obj_code][scale] = {"per_query_gt_inds": per_query_gt_inds.detach().cpu().numpy().tolist(), 
                                                          "query_matched_mask": query_matched_mask.detach().cpu().numpy().tolist()}
            else:
                self.assignments_save[obj_code] = {scale: {"per_query_gt_inds": per_query_gt_inds.detach().cpu().numpy().tolist(), 
                                                          "query_matched_mask": query_matched_mask.detach().cpu().numpy().tolist()}
                                                          }
        
    def single_output_forward(self, outputs, targets):
        outputs["hand_model_pose"] = self.get_hand_model_pose(outputs)
        if targets["assignment"][0] != None:
            assignments_tensor = {"per_query_gt_inds":[], "query_matched_mask": []}
            for i in range(len(targets["assignment"])):
                assignments_tensor["per_query_gt_inds"].append(torch.tensor(targets["assignment"][i]["per_query_gt_inds"]))
                assignments_tensor["query_matched_mask"].append(torch.tensor(targets["assignment"][i]["query_matched_mask"]))
            assignments_tensor["per_query_gt_inds"] = torch.stack(assignments_tensor["per_query_gt_inds"], dim=0)
            assignments_tensor["query_matched_mask"] = torch.stack(assignments_tensor["query_matched_mask"], dim=0)
            assignments = assignments_tensor
        else:
            assignments = self.matcher(outputs, targets)
            self.assignment_saving(assignments, targets)

        matched_preds, matched_targets = self.get_matched_by_assignment(outputs, targets, assignments)
        outputs["matched"] = matched_preds
        targets["matched"] = matched_targets
        outputs["hand"] = self.get_hand(outputs["matched"]["hand_model_pose"], targets["obj_pc"], assignments)
        targets["hand"] = self.get_hand(targets["matched"]["hand_model_pose"], targets["obj_pc"], assignments)

        losses = {}
        for name, weight in self.loss_weights.items():
            if hasattr(self, f"get_{name}_loss"):
                m = getattr(self, f"get_{name}_loss")
                _loss_dict = m(outputs, targets, assignments)
                losses.update(_loss_dict)
            else:
                available_loss = [x[4:] for x in dir(self) if x.endswith("_loss") and not x.startswith("_")]
                raise NotImplementedError(f"Unable to calculate {name} loss. Available losses: {available_loss}")
        return losses
    
    def matchfree_output_forward(self, outputs, targets):
        outputs["hand_model_pose"] = self.get_hand_model_pose(outputs)

        targets["translation_norm"] = targets["norm_pose"][..., :3]
        targets["qpos_norm"] = targets["norm_pose"][..., 3:25]
        targets["rotation"] = targets["norm_pose"][..., 25:]

        outputs["matched"] = outputs
        targets["matched"] = targets

        outputs["hand"] = self.get_hand(outputs["matched"]["hand_model_pose"], targets["obj_pc"])
        targets["hand"] = self.get_hand(targets["matched"]["hand_model_pose"], targets["obj_pc"])
        losses = {}
        for name, weight in self.loss_weights.items():
            if hasattr(self, f"get_{name}_loss"):
                m = getattr(self, f"get_{name}_loss")
                _loss_dict = m(outputs, targets, None)
                losses.update(_loss_dict)
            else:
                available_loss = [x[4:] for x in dir(self) if x.endswith("_loss") and not x.startswith("_")]
                raise NotImplementedError(f"Unable to calculate {name} loss. Available losses: {available_loss}")
        return {"outputs": losses}
    
    def forward(self, outputs, targets):
        loss_dict = {}
        loss_dict["outputs"] = self.single_output_forward(outputs["outputs"], targets)

        # calculate loss for intermediate outputs
        if "aux_outputs" in outputs:
            num_intermediate = len(outputs["aux_outputs"])
            loss_dict["aux_outputs"] = [deepcopy({}) for _ in range(num_intermediate)]
            for k in range(num_intermediate):
                interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )
                for interm_key in interm_loss_dict:
                    loss_dict["aux_outputs"][k][interm_key] = interm_loss_dict[interm_key]
        return loss_dict

    def get_matched_by_assignment(
        self,
        predictions: Dict,
        targets: Dict,
        assignment: Dict,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            matched_preds: Dict[str, Tensor of size (K, D)]
            matched_targets: Dict[str, Tensor of size (K, D)]
        """
        per_query_gt_inds = assignment["per_query_gt_inds"] # (B, num_queries)
        query_matched_mask = assignment["query_matched_mask"]  # (B, num_queries)
        K = query_matched_mask.long().sum()  # K = number of matched
        B = predictions["features"].size(0)
        matched_preds, matched_targets = {}, {}
        pred_target_match_key_map = {
            "qpos_norm": "norm_pose",
            "rotation": "norm_pose",
            "translation_norm": "norm_pose",
            "hand_model_pose": "hand_model_pose",
        }
        pose_slices = {
            "translation_norm": (0, 3),
            "qpos_norm": (3, 25),
            "rotation": (25, targets["norm_pose"][0].size(-1)),
        }

        for pred_key, target_key in pred_target_match_key_map.items():
            if pred_key not in predictions.keys():
                continue
            pred = predictions[pred_key]
            target = targets[target_key]
            matched_pred_buffer = []
            matched_target_buffer = []
            for i in range(B):
                _matched_pred, _matched_target = self._get_matched(
                    pred[i],
                    target[i],
                    per_query_gt_inds[i],
                    query_matched_mask[i],
                )
                matched_pred_buffer.append(_matched_pred)
                matched_target_buffer.append(_matched_target)
            matched_preds[pred_key] = torch.cat(matched_pred_buffer, dim=0)
            matched_targets[pred_key] = torch.cat(matched_target_buffer, dim=0)
            if pred_key in pose_slices.keys():
                _s, _e = pose_slices[pred_key]
                matched_targets[pred_key] = matched_targets[pred_key][:, _s:_e]
            assert K == matched_preds[pred_key].size(0)
            assert K == matched_targets[pred_key].size(0)
        return matched_preds, matched_targets

    def get_hand(self, hand_model_pose: Tensor, point_cloud: Tensor, assignment: Dict) -> Dict:
        K = hand_model_pose.size(0)
        B = point_cloud.size(0)
        query_matched_mask = assignment["query_matched_mask"].long().sum(-1)  # (B, )
        batch_point_cloud = []
        for i in range(B):
            _pc = point_cloud[i:i + 1].expand(query_matched_mask[i].item(), -1, -1)
            batch_point_cloud.append(_pc)
        batch_point_cloud = torch.cat(batch_point_cloud, dim=0)  # (K, N, 3)
        assert batch_point_cloud.size(0) == K
        hand = self.hand_model(
            hand_model_pose,
            batch_point_cloud,
            with_penetration=True,
            with_surface_points=True,
            with_penetration_keypoints=True,
            with_contact_candidates=True
        )
        return hand

    def get_hand_model_pose(self, pred_dict):
        pred_qpos_norm = pred_dict["qpos_norm"]
        pred_rotation = pred_dict["rotation"]
        pred_translation_norm = pred_dict["translation_norm"]

        def _unnormalize(normed, minmax):
            _min = minmax[..., 0]
            _max = minmax[..., 1]
            return normed * (_max - _min) + _min

        tranlation = _unnormalize(pred_translation_norm, self.t_minmax)
        qpos = _unnormalize(pred_qpos_norm, self.q_minmax)
        # rotation to axis angle
        rotation_type = pred_dict["rotation_type"]
        to_axisangle = getattr(Rot2Axisangle, f"{rotation_type}2axisangle")
        axisangle = to_axisangle(pred_rotation)

        hand_model_pose = torch.cat([tranlation, axisangle, qpos], dim=-1)
        return hand_model_pose

    def get_translation_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        # reconstruction loss on translation params
        pred = prediction["matched"]["translation_norm"]
        target = target["matched"]["translation_norm"]
        loss = {"translation": self._get_regression_loss(pred, target)}
        return loss

    def get_qpos_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        # reconstruction loss on qpos params
        pred = prediction["matched"]["qpos_norm"]
        target = target["matched"]["qpos_norm"]
        loss = {"qpos": self._get_regression_loss(pred, target)}
        return loss

    def get_rotation_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        rotation_type = prediction["rotation_type"]
        pred = prediction["matched"]["rotation"]
        target = target["matched"]["rotation"]
        if hasattr(self, f"_get_{rotation_type}_loss"):
            m = getattr(self, f"_get_{rotation_type}_loss")
            loss = m(pred, target)
            return loss
        else:
            raise NotImplementedError(f"Unable to calculate {rotation_type} loss.")

    def get_hand_chamfer_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        # chamfer loss between predict-hand point cloud and target-hand point cloud
        pred_hand_pc = prediction["hand"]["surface_points"]
        target_hand_pc = target["hand"]["surface_points"]
        chamfer_loss = chamfer_distance(pred_hand_pc, target_hand_pc, point_reduction="sum", batch_reduction="mean")[0]
        loss = {"hand_chamfer": chamfer_loss}
        return loss

    def get_cls_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        logits = prediction["logit"].flatten()  # (B x N, )
        labels = assignment["query_matched_mask"].flatten().float()  # (B x N, )
        loss = {"cls": F.binary_cross_entropy_with_logits(logits, labels)}
        return loss
    
    def get_distance_loss(self, prediction, target, assignment, thres_dis=0.01):
        # loss_dis
        dis_pred = prediction["hand"]['contact_candidates_dis']
        small_dis_pred = dis_pred < thres_dis ** 2
        loss_dis = dis_pred[small_dis_pred].sum() / dis_pred.size(0)
        loss = {"distance": loss_dis}
        return loss
    
    def get_obj_penetration_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        # batch_size, num_queries = prediction["features"].shape[:2]
        # signed squared distances from object_pc to hand, inside positive, outside negative
        distances = prediction["hand"]["penetration"]  # (B * num_queries, num_object_points)
        # loss_pen
        loss_pen = distances[distances > 0].sum() / (distances.shape[0])
        loss = {"obj_penetration": loss_pen}
        return loss

    def get_self_penetration_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        # batch_size, num_queries = prediction["features"].shape[:2]
        # loss_spen
        penetration_keypoints = prediction["hand"]["penetration_keypoints"]
        dis_spen = (penetration_keypoints.unsqueeze(1) - penetration_keypoints.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
        dis_spen = 0.02 - dis_spen
        dis_spen[dis_spen < 0] = 0
        loss_spen = dis_spen.sum() / (penetration_keypoints.shape[0])
        loss = {"self_penetration": loss_spen}
        return loss

    def get_query_diff_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        loss = {"query_diff": cosine_loss(prediction["features"])}
        return loss

    def get_pose_diff_loss(self, prediction, target, assignment) -> Dict[str, Tensor]:
        qpos = prediction["qpos_norm"]
        rotation = prediction["rotation"]
        translation = prediction["translation_norm"]
        loss_rt = .5 * cosine_loss(rotation) + .5 * cosine_loss(translation)
        loss_qpos = cosine_loss(qpos)
        loss = {"pose_diff": torch.min(loss_rt, loss_qpos)}
        return loss

    def _get_euler_loss(self, prediction, target) -> Dict[str, Tensor]:
        """
        Params:
        prediction: A tensor of size (K, 3)
        target: A tensor of size (K, 3)
        """
        error = (prediction - target).abs()
        error = torch.where(error < 0.5, error, 1 - error).sum(-1).mean()
        loss = {"rotation": error}
        return loss

    def _get_quaternion_loss(self, prediction, target) -> Dict[str, Tensor]:
        """
        Params:
        prediction: A tensor of size (K, 4)
        target: A tensor of size (K, 4)
        """
        loss = {"rotation": (1.0 - (prediction * target).sum(-1).abs()).mean()}
        return loss

    def _get_rotation_6d_loss(self, prediction, target) -> Dict[str, Tensor]:
        """
        Params:
        prediction: A tensor of size (K, 6)
        target: A tensor of size (K, 6)
        """
        loss = {"rotation": self._get_regression_loss(prediction, target)}
        return loss

    def _get_matched(self, pred, gt, gt_inds, matched_mask) -> Tuple[Tensor, Tensor]:
        """
        Params:
        pred: A tensor of size (N, D)
        gt: A tensor of size (M, D)
        gt_inds: A tensor of size (N, )
        matched_mask: A tensor of size (N, )
        Return:
        matched_pred: A tensor of size (K, D), where K = sum(matched_mask)
        matched_gt: A tensor of size (K, D), where K = sum(matched_gt)
        """
        matched_pred = pred[matched_mask == 1, :]
        matched_gt = gt[gt_inds, :][matched_mask == 1, :]
        return matched_pred, matched_gt

    def _get_regression_loss(
        self,
        prediction: Tensor,
        target: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Params:
        prediction: A tensor of size (K, D)
        target: A tensor of size (K, D)
        """
        loss = huber_loss(prediction - target).sum(-1).mean()
        return loss


def cosine_loss(feature, norm=True):
    """
    args:
        feature: (B, N, C)
    """
    if norm:
        normalized_feature = F.normalize(feature, p=2, dim=-1)
    else:
        normalized_feature = feature
    cosine_similarities = torch.bmm(normalized_feature, normalized_feature.transpose(-1, -2))
    diagonal_mask = torch.eye(cosine_similarities.shape[-1]).unsqueeze(0).to(feature.device)
    cosine_similarities *= 1 - diagonal_mask
    loss = cosine_similarities.abs().sum() / ((1 - diagonal_mask).sum() * feature.size(0))  # mean
    return loss


def huber_loss(error, delta=1.0):
    """
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss
