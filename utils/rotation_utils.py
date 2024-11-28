import math
import torch
import torch.nn.functional as F
import pytorch3d.transforms as T
from torch.functional import Tensor


class EulerConverter:
    """
    A class for converting euler angles with axes = 'sxyz' (as defined in transforms3d)
    to other rotation representations.

    Support batch operations.

    Expect tensor of shape (..., 3) as input.

    All types of outputs can be transformed to rotation matrices, which
    are identical to the results of transforms3d.euler.euler2mat(euler, axes='sxyz'),
    by functions like pytorch3d.transforms.XXX_to_matrix().
    """

    def to_euler(self, euler):
        return euler

    def to_matrix(self, euler):
        return T.axis_angle_to_matrix(
            self.to_axisangle(euler)
        )

    def to_rotation_6d(self, euler):
        return T.matrix_to_rotation_6d(
            self.to_matrix(euler),
        )

    def to_quaternion(self, euler):
        return T.axis_angle_to_quaternion(
            self.to_axisangle(euler)
        )

    def to_axisangle(self, euler):
        return torch.flip(
            T.matrix_to_axis_angle(T.euler_angles_to_matrix(euler, "ZYX")),
            dims=[-1]
        )


class RotNorm:
    """
    A class for normalizing rotation representations
    """
    @staticmethod
    def norm_euler(euler: Tensor) -> Tensor:
        """
        euler: A tensor of size: (B, 3, N)
        """
        lower_bounds = torch.ones_like(euler) * math.pi * -1.0
        upper_bounds = torch.ones_like(euler) * math.pi
        return (euler - lower_bounds) / (upper_bounds - lower_bounds)

    @staticmethod
    def norm_quaternion(quaternion: Tensor) -> Tensor:
        """
        quaternion: A tensor of size: (B, 4, N)
        """
        return F.normalize(quaternion)

    @staticmethod
    def norm_rotation_6d(rot6d: Tensor) -> Tensor:
        """
        rot6d: A tensor of size: (B, 6, N)
        """
        vector_1 = F.normalize(rot6d[:, :3, :])
        vector_2 = F.normalize(rot6d[:, 3:6, :])
        return torch.cat([vector_1, vector_2], dim=1)

    def norm_other(tensor) -> Tensor:
        return tensor


class Rot2Axisangle:
    """
    A class for converting rotation representations to axisangle
    """
    @staticmethod
    def euler2axisangle(euler):
        return torch.flip(
            T.matrix_to_axis_angle(T.euler_angles_to_matrix(euler, "ZYX")),
            dims=[-1]
        )

    @staticmethod
    def quaternion2axisangle(quaternion):
        return T.quaternion_to_axis_angle(quaternion)

    @staticmethod
    def rotation_6d2axisangle(rot6d):
        B, N = rot6d.shape[:2]
        mat = robust_compute_rotation_matrix_from_ortho6d(rot6d.reshape(B * N, 6))
        return T.matrix_to_axis_angle(mat.reshape(B, N, 3, 3))

    @staticmethod
    def matrix2axisangle(mat):
        return T.matrix_to_axis_angle(mat)

    @staticmethod
    def axisangle2axisangle(axisangle):
        return axisangle


# Codes borrowed from dexgraspnet
def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out
