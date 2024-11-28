
import sys
sys.path.append("./third_party/pointnet2/")

import torch.nn as nn
from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from torch.functional import Tensor

# class Pointnet2Backbone(nn.Module):
#     """
#        Backbone network for point cloud feature learning.
#        Based on Pointnet++ single-scale grouping network.

#        Parameters
#        ----------
#        input_feature_dim: int
#             Number of input channels in the feature descriptor for each point.
#             e.g. 3 for RGB.
#     """

#     def __init__(self, cfg):
#         super().__init__()

#         self.sa1 = PointnetSAModuleVotes(
#             npoint=cfg.layer1.npoint,
#             radius=cfg.layer1.radius_list[0],
#             nsample=cfg.layer1.nsample_list[0],
#             mlp=cfg.layer1.mlp_list,
#             use_xyz=cfg.use_xyz,
#             normalize_xyz=cfg.normalize_xyz
#         )

#         self.sa2 = PointnetSAModuleVotes(
#             npoint=cfg.layer2.npoint,
#             radius=cfg.layer2.radius_list[0],
#             nsample=cfg.layer2.nsample_list[0],
#             mlp=cfg.layer2.mlp_list,
#             use_xyz=cfg.use_xyz,
#             normalize_xyz=cfg.normalize_xyz
#         )

#         self.sa3 = PointnetSAModuleVotes(
#             npoint=cfg.layer3.npoint,
#             radius=cfg.layer3.radius_list[0],
#             nsample=cfg.layer3.nsample_list[0],
#             mlp=cfg.layer3.mlp_list,
#             use_xyz=cfg.use_xyz,
#             normalize_xyz=cfg.normalize_xyz
#         )
#         self.fp1 = PointnetFPModule(mlp=[cfg.layer3.mlp_list[-1] + cfg.layer2.mlp_list[-1], cfg.layer3.mlp_list[-1], cfg.layer3.mlp_list[-1]])
#         self.fp2 = PointnetFPModule(mlp=[cfg.layer3.mlp_list[-1] + cfg.layer1.mlp_list[-1], cfg.layer3.mlp_list[-1], cfg.layer3.mlp_list[-1]])

#     def _break_up_pc(self, pc):
#         xyz = pc[..., 0:3].contiguous()
#         features = (
#             pc[..., 3:].transpose(1, 2).contiguous()
#             if pc.size(-1) > 3 else None
#         )
#         return xyz, features

#     def forward(self, pointcloud: Tensor):
#         """
#             Forward pass of the network

#             Parameters
#             ----------
#             pointcloud: Variable(Tensor)
#                 (B, N, 3 + input_feature_dim) tensor
#                 Point cloud to run predicts on
#                 Each point in the point-cloud MUST
#                 be formated as (x, y, z, features...)

#             Returns
#             ----------
#                 xyz: float32 Tensor of shape (B, K, 3)
#                 features: float32 Tensor of shape (B, D, K)
#                 inds: int64 Tensor of shape (B, K) values in [0, N-1]
#         """
#         xyz, features = self._break_up_pc(pointcloud)
#         xyz1, features1, fps_inds = self.sa1(xyz, features)
#         xyz2, features2, fps_inds = self.sa2(xyz1, features1)
#         xyz3, features3, fps_inds = self.sa3(xyz2, features2)
#         features = self.fp1(xyz2, xyz3, features2, features3)
#         features = self.fp2(xyz1, xyz2, features1, features)

#         return xyz1, features

class Pointnet2Backbone(nn.Module):
    """
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, cfg):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=cfg.layer1.npoint,
            radius=cfg.layer1.radius_list[0],
            nsample=cfg.layer1.nsample_list[0],
            mlp=cfg.layer1.mlp_list,
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=cfg.layer2.npoint,
            radius=cfg.layer2.radius_list[0],
            nsample=cfg.layer2.nsample_list[0],
            mlp=cfg.layer2.mlp_list,
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=cfg.layer3.npoint,
            radius=cfg.layer3.radius_list[0],
            nsample=cfg.layer3.nsample_list[0],
            mlp=cfg.layer3.mlp_list,
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: Tensor):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(Tensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
                xyz: float32 Tensor of shape (B, K, 3)
                features: float32 Tensor of shape (B, D, K)
                inds: int64 Tensor of shape (B, K) values in [0, N-1]
        """
        xyz, features = self._break_up_pc(pointcloud)
        xyz, features, fps_inds = self.sa1(xyz, features)
        xyz, features, fps_inds = self.sa2(xyz, features)
        xyz, features, fps_inds = self.sa3(xyz, features)

        return xyz, features
