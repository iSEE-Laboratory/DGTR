from functools import partial
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.functional import Tensor


class Matcher(nn.Module):
    def __init__(self, weight_dict: Dict[str, float]):
        super().__init__()
        # exclude weights equaling 0
        self.weight_dict = {k: v for k, v in weight_dict.items() if v > 0}

    @torch.no_grad()
    def forward(self, preds: Dict, targets: Dict):
        _example = preds["rotation"]
        device = _example.device
        batch_size, nqueries = _example.shape[:2]
        rotation_type = targets["rotation_type"][0]

        cost_matrices = []
        for name, weight in self.weight_dict.items():
            m = getattr(self, f"get_{name}_cost_mat")
            if name == "rotation":
                cost_mat = m(preds, targets, weight=weight, rotation_type=rotation_type)
            cost_mat = m(preds, targets, weight=weight)
            cost_matrices.append(cost_mat)
        final_cost = [sum(x).detach().cpu().numpy() for x in zip(*cost_matrices)]

        assignments = []
        # auxiliary variables useful for batched loss computation
        per_query_gt_inds = torch.zeros(
            [batch_size, nqueries], dtype=torch.int64, device=device
        )
        query_matched_mask = torch.zeros(
            [batch_size, nqueries], dtype=torch.float32, device=device
        )
        for b in range(batch_size):
            assign = []
            assign = linear_sum_assignment(final_cost[b])
            assign = [
                torch.from_numpy(x).long().to(device)
                for x in assign
            ]
            per_query_gt_inds[b, assign[0]] = assign[1]
            query_matched_mask[b, assign[0]] = 1
            assignments.append(assign)
        return {
            "per_query_gt_inds": per_query_gt_inds,
            "query_matched_mask": query_matched_mask,
        }

    def get_hand_mesh_cost_mat(
        self,
        prediction: Tensor,
        targets: List[Tensor],
        weight: float = 1.0,
    ) -> List[Tensor]:
        # TODO: implement chamfer loss for hand mesh cost
        raise NotImplementedError("Unable to calculate hand mesh cost matrix yet. Please help me to implement it ^_^")

    def get_qpos_cost_mat(
        self,
        prediction: Tensor,
        targets: List[Tensor],
        weight: float = 1.0,
    ) -> List[Tensor]:
        pred_qpos = prediction["qpos_norm"]
        target_qpos = [x[..., 3:25] for x in targets["norm_pose"]]
        return self._get_cost_mat_by_elementwise(pred_qpos, target_qpos, weight=weight)

    def get_translation_cost_mat(
        self,
        prediction: Tensor,
        targets: List[Tensor],
        weight: float = 1.0,
    ) -> List[Tensor]:
        pred_translation = prediction["translation_norm"]
        target_translation = [x[..., :3] for x in targets["norm_pose"]]
        return self._get_cost_mat_by_elementwise(pred_translation, target_translation, weight=weight)

    def get_rotation_cost_mat(
        self,
        prediction: Tensor,
        targets: List[Tensor],
        weight: float = 1.0,
        rotation_type: str = "quaternion",
    ) -> List[Tensor]:
        if hasattr(self, f"_get_{rotation_type}_cost_mat"):
            m = getattr(self, f"_get_{rotation_type}_cost_mat")
            pred_rotation = prediction["rotation"]  # (num_queries, D)
            target_rotation = [x[..., 25:] for x in targets["norm_pose"]]  # [(ngt1, D), (ngt2, D), ...]
            return m(pred_rotation, target_rotation, weight)
        else:
            raise NotImplementedError(f"Unable to get {rotation_type} cost matrix")

    def _get_cost_mat_by_elementwise(
        self,
        prediction: Tensor,
        targets: List[Tensor],
        weight: float = 1.0,
        element_wise_func: Callable[[Tensor, Tensor], Tensor] = partial(F.l1_loss, reduction="none"),
    ) -> List[Tensor]:
        """
        calculate cost matrix by element-wise operations

        Params:
        prediction: B, nqueries, D
        targets: [(ngt1, D), (ngt2, D), ...]
        weight: a float number for current cost matrix
        element_wise_func: an element-wise function for two tensors. Default is l1_loss

        return:
        cost_mat: [(nqueries, ngt1), (nqueries, ngt2), ...]
        """
        B = prediction.size(0)
        assert B == len(targets), f"batch size and len(targets) should be the same"
        nqueries = prediction.size(1)
        cost_mat = []
        for i in range(B):
            rot, gt = prediction[i], targets[i]
            ngt = gt.size(0)
            rot = rot.unsqueeze(1).expand(-1, ngt, -1)  # (nqueries, ngt, D)
            gt = gt.unsqueeze(0).expand(nqueries, -1, -1)  # (nqueries, ngt, D)
            cost = element_wise_func(rot, gt).sum(-1)
            cost_mat.append(weight * cost)
        return cost_mat

    def _get_quaternion_cost_mat(self, prediction: Tensor, targets: List[Tensor], weight: float = 1.0) -> List[Tensor]:
        B = prediction.size(0)
        assert B == len(targets), f"batch size and len(targets) should be the same"
        cost_mat = []
        for i in range(B):
            rot, gt = prediction[i], targets[i]
            cost = 1 - (rot @ gt.T).abs().detach()
            cost_mat.append(weight * cost)
        return cost_mat

    def _get_rotation_6d_cost_mat(self, prediction: Tensor, targets: List[Tensor], weight: float = 1.0) -> List[Tensor]:
        cost_mat = self._get_cost_mat_by_elementwise(
            prediction,
            targets,
            weight=weight,
        )
        return cost_mat

    def _get_euler_cost_mat(self, prediction: Tensor, targets: List[Tensor], weight: float = 1.0) -> List[Tensor]:
        """
        specially-designed l1 loss for euler angles
        """
        B = prediction.size(0)
        assert B == len(targets), f"batch size and len(targets) should be the same"
        cost_mat = []
        for i in range(B):
            rot, gt = prediction[i].unsqueeze(1), targets[i].unsqueeze(0)
            error = (rot - gt).abs().sum(-1)
            cost = torch.where(error < 0.5, error, 1 - error)
            cost_mat.append(weight * cost)
        return cost_mat
