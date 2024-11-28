import glob
import json
import os.path as osp
from typing import List

import h5py
import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data import Dataset
import pytorch3d.transforms as T

from utils.rotation_utils import EulerConverter, RotNorm


class DgnBase(Dataset):
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0',
    ]
    rotation_names = ['WRJRx', 'WRJRy', 'WRJRz']
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']

    q_minmax = torch.tensor(
        [[-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
         [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
         [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
         [0.0, 0.785], [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
         [-1.047, 1.047], [0.0, 1.222], [-0.209, 0.209], [-0.524, 0.524], [-1.571, 0.0]]
    )
    t_minmax = torch.tensor([[-0.22, 0.22], [-0.22, 0.22], [-0.22, 0.22]])
    r_minmax = torch.tensor([[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]])

    def __init__(self, rotation_type: str = 'euler') -> None:
        super().__init__()
        # default rotation type is "euler" and axes are 'sxyz'
        rot_transform = EulerConverter()
        rot_norm = RotNorm()
        if not hasattr(rot_transform, f'to_{rotation_type}'):
            raise NotImplementedError(f'Unsupported rotation type: {rotation_type}')
        else:
            self.rot_transform = getattr(rot_transform, f'to_{rotation_type}')
            self.rot_norm = getattr(rot_norm, f'norm_{rotation_type}')
        self.euler_to_axisangle = rot_transform.to_axisangle
        self.rotation_type = rotation_type

    @staticmethod
    def _norm_by_minmax(input: Tensor, minmax: Tensor) -> Tensor:
        """
        input: (N, C)
        minmax: (N, 2)
        """
        normed = (input - minmax[:, 0]) / (minmax[:, 1] - minmax[:, 0])
        return normed
    
    @staticmethod
    def _denorm_by_minmax(input: Tensor, minmax: Tensor) -> Tensor:
        """
        input: (N, C)
        minmax: (N, 2)
        """
        denormed = input * (minmax[..., 1] - minmax[..., 0]) + minmax[..., 0]
        return denormed


    @staticmethod
    def collate_fn(batch):
        input_dict = {}
        for k in batch[0]:
            # if k in ["obj_pc", "convex_hull"]:
            if k == "obj_pc":
                input_dict[k] = torch.stack([sample[k] for sample in batch])
            elif k == "convex_hull":
                input_dict[k] = torch.cat([sample[k] for sample in batch], dim=0)
            else:
                input_dict[k] = [sample[k] for sample in batch]
        return input_dict

class DgnSetFull(DgnBase):
    """
    Dataset class for all DexGraspNet objects and poses
    __getitem__() returns one object point cloud and several poses at a time
    One object only have one scale and the corresponding poses.
    """

    def __init__(
        self,
        data_root: str,
        object_code_list: List[str],
        num_points: int = 4096,
        rotation_type: str = "quaternion",
        num_gt: int = 9999,
        is_train = True,
        assignment_path = None
    ) -> None:
        super().__init__(rotation_type=rotation_type)
        self.data_root = data_root
        self.num_points = num_points
        self.data_path = object_code_list
        self.num_gt = num_gt

        self.is_train = is_train
        self.assignment_path = assignment_path
        self._load_data()

    def _load_data(self):
        print(self.data_path)
        
        with h5py.File(self.data_path, 'r') as hf:
            self.data = [{'obj_pc': torch.tensor(np.array(hf[f'obj_{i}']), dtype=torch.float),
                     'hand_pose': torch.tensor(np.array(hf[f'obj_{i}'].attrs['hand_pose']), dtype=torch.float),
                     'obj_code': hf[f'obj_{i}'].attrs['obj_code'],
                     'scale': hf[f'obj_{i}'].attrs['scale']} for i in range(len(hf))]
            print(len(hf))
            
        if self.assignment_path:
            with open(self.assignment_path) as rf:
                self.assignment = json.load(rf)
            print("Loading assignment")
        else:
            print("Do not load assignment")

    def __getitem__(self, index):
        obj_code = self.data[index]['obj_code']
        hand_poses = self.data[index]['hand_pose']
        scale = self.data[index]['scale']
        obj_point_cloud = self.data[index]['obj_pc']

        if self.is_train and self.assignment_path:
            assignment = self.assignment[obj_code][str(scale)]
        else:
            assignment = None

        # pose: n,28
        hand_qpos = hand_poses[..., :-6]
        hand_euler = hand_poses[..., -6:-3]
        hand_translation = hand_poses[..., -3:]

        hand_rotation = self.euler_to_axisangle(hand_euler)
        hand_poses = torch.cat([hand_translation, hand_rotation, hand_qpos], dim=-1)

        # object
        obj_point_cloud *= scale

        # norm
        norm_qpos = self._norm_by_minmax(hand_qpos, self.q_minmax)
        norm_rotation = self.rot_transform(hand_euler)
        if self.rotation_type == "euler":
            norm_rotation = self._norm_by_minmax(norm_rotation, self.r_minmax)
        norm_translation = self._norm_by_minmax(hand_translation, self.t_minmax)
        norm_poses = torch.cat([norm_translation, norm_qpos, norm_rotation], dim=-1)
        ret_dict = {
            "obj_pc": obj_point_cloud,  # (N, 3)
            "scale": scale,
            "obj_code": obj_code,
            "rotation_type": self.rotation_type,
            "norm_pose": norm_poses[:self.num_gt],  # shape: (num_grasps, trans(3) + pose(22) + specified_rotation(?))
            "hand_model_pose": hand_poses[:self.num_gt],  # shape: (num_grasps, trans(3) + axisangle(?) + pose(22))
            "assignment": assignment
        }
        return ret_dict

    def __len__(self,):
        return len(self.data)
    
