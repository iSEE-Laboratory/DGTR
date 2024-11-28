"""
Codes borrowed from https://github.com/PKU-EPIC/DexGraspNet/blob/main/grasp_generation/utils/initializations.py
"""

import argparse
import math
import os
import os.path as osp
import random
from typing import List

import numpy as np
import plotly.graph_objects as go
import pytorch3d.ops
import pytorch3d.structures
import pytorch3d.transforms as T
import torch
import torch.nn.functional
import transforms3d
import trimesh as tm
from torch.functional import Tensor
from tqdm import trange

from datasets import DgnSetFull


def init_grasps(
    obj_point_cloud: Tensor,
    obj_points: Tensor,
    convex_hull: Tensor,
    cfg,
) -> Tensor:
    """
    Params:
        obj_point_cloud: full object point cloud  (B, N, 3)
        obj_points: selected object points to init grasps  (B, M, 3)
        convex_hull: inflated convex hull of the object (follow DexGraspNet)  (B, K, 3)
    Returns:
        hand_poses: (B, M, translation(3) + rotation_dim + joint-angles(22))
    """
    device = obj_point_cloud.device
    B, n_grasps = obj_points.shape[:2]

    rotation = torch.zeros((B, n_grasps, 3, 3), dtype=torch.float, device=device)
    translation = torch.zeros((B, n_grasps, 3), dtype=torch.float, device=device)
    closest_on_hull = pytorch3d.ops.knn_points(obj_points, convex_hull, K=1, return_nn=True)[2].squeeze()  # (B, n_grasps, 3)
    n = obj_points - closest_on_hull
    n /= n.norm(dim=1).unsqueeze(1)  # (B, n_grasps, 3)

    # sample randomly
    distance = cfg.distance_lower + (cfg.distance_upper - cfg.distance_lower) * torch.rand([B, n_grasps, 1], dtype=torch.float, device=device)
    deviate_theta = cfg.theta_lower + (cfg.theta_upper - cfg.theta_lower) * torch.rand([B, n_grasps], dtype=torch.float, device=device)
    process_theta = 2 * math.pi * torch.rand([B, n_grasps], dtype=torch.float, device=device)
    rotate_theta = 2 * math.pi * torch.rand([B, n_grasps], dtype=torch.float, device=device)
    euler_local = torch.stack([process_theta, deviate_theta, rotate_theta], dim=-1)  # (B, n_grasps, 3)
    euler_global = torch.stack([
        torch.atan2(n[..., 1], n[..., 0]).to(device) - torch.pi / 2,
        -torch.acos(n[..., 2]).to(device),
        torch.zeros_like(rotate_theta),
    ], dim=-1)  # (B, n_grasps, 3)

    # rotation
    rotation_local = T.euler_angles_to_matrix(euler_local, "ZXZ")
    rotation_global = T.euler_angles_to_matrix(euler_global, "ZXZ")  # (B, n_grasps, 3, 3)
    rotation_hand = T.euler_angles_to_matrix(torch.tensor([[0, -np.pi / 3, 0]], dtype=torch.float, device=device), "ZXZ").view(1, 1, 3, 3)
    rotation = rotation_global @ rotation_local @ rotation_hand  # (B, n_grasps, 3, 3)
    rotation_map = {
        "euler": "euler_angles",
        "axisangle": "axis_angle",
    }
    rotation_type = rotation_map.get(cfg.rotation_type, cfg.rotation_type)
    rotation_transform = getattr(T, f"matrix_to_{rotation_type}")
    rotation = rotation_transform(rotation)  # (B, n_grasps, rotation_dim)
    # translation
    hand_base_direction = torch.tensor([0, 0, 1], dtype=torch.float, device=device).view(1, 1, -1, 1)
    translation = closest_on_hull - 0.2 * distance * (rotation_global @ rotation_local @ hand_base_direction).squeeze(-1)

    # qpos (joint angles)
    joint_angles_mu = torch.tensor([
        0.1, 0.0, 0.6, 0.0, 0.0,
        0.0, 0.6, 0.0, -0.1, 0.0,
        0.6, 0.0, 0.0, -0.2, 0.0, 0.6,
        0.0, 0.0, 1.2, 0.0, -0.2, 0.0
    ], dtype=torch.float, device=device)
    n_dofs = len(DgnSetFull.joint_names)
    joints_upper = DgnSetFull.q_minmax[:, 1].to(device)
    joints_lower = DgnSetFull.q_minmax[:, 0].to(device)
    joint_angles_sigma = cfg.jitter_strength * (joints_upper - joints_lower)
    joint_angles = torch.empty([B, n_grasps, n_dofs], dtype=torch.float, device=device)
    for i in range(n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, :, i],
            joint_angles_mu[i],
            joint_angles_sigma[i],
            joints_lower[i] - 1e-6,
            joints_upper[i] + 1e-6
        )
    # assembly hand pose
    hand_poses = torch.cat([
        translation,
        rotation,
        joint_angles,
    ], dim=-1)  # (B, n_grasps, 3 + rotation_dim + 22)

    return hand_poses


def dgn_init_grasps(
    hand_model,
    object_meshes: List[tm.Trimesh],
    scales: List[float],
    n_grasps: int,
    args
):
    """
    Initialize grasp translation, rotation, joint angles, and contact point indices

    Parameters
    ----------
    hand_model: hand_model.HandModel
    object_meshes: List[tm.Trimesh]
    args: Namespace
    """

    device = hand_model.device
    n_objects = len(object_meshes)

    # initialize translation and rotation

    translation = torch.zeros((n_objects, n_grasps, 3), dtype=torch.float, device=device)
    rotation = torch.zeros((n_objects, n_grasps, 3, 3), dtype=torch.float, device=device)

    convex_hulls = []
    ps, closest = [], []
    for i in trange(n_objects, desc="Init convex hull", leave=True):

        # get inflated convex hull

        mesh_convex_hull = object_meshes[i].convex_hull
        vertices = mesh_convex_hull.vertices.copy()
        faces = mesh_convex_hull.faces
        vertices *= scales[i]
        mesh_convex_hull = tm.Trimesh(vertices, faces)
        mesh_convex_hull.faces = mesh_convex_hull.faces[mesh_convex_hull.remove_degenerate_faces()]
        vertices += 0.2 * 10 * scales[i] * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh_inflated_convex_hull = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        convex_hulls.append(mesh_inflated_convex_hull)
        vertices = torch.tensor(mesh_inflated_convex_hull.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh_inflated_convex_hull.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # sample points

        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * n_grasps)
        p = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=n_grasps)[0][0]
        closest_points, _, _ = mesh_convex_hull.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)
        ps.append(p)
        closest.append(closest_points)
        # sample parameters

        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([n_grasps], dtype=torch.float, device=device)
        deviate_theta = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([n_grasps], dtype=torch.float, device=device)
        process_theta = 2 * math.pi * torch.rand([n_grasps], dtype=torch.float, device=device)
        rotate_theta = 2 * math.pi * torch.rand([n_grasps], dtype=torch.float, device=device)
        # solve transformation
        # rotation_hand: rotate the hand to align its grasping direction with the +z axis
        # rotation_local: jitter the hand's orientation in a cone
        # rotation_global and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull

        rotation_local = torch.zeros([n_grasps, 3, 3], dtype=torch.float, device=device)
        rotation_global = torch.zeros([n_grasps, 3, 3], dtype=torch.float, device=device)
        for j in range(n_grasps):
            rotation_local[j] = torch.tensor(
                transforms3d.euler.euler2mat(process_theta[j], deviate_theta[j], rotate_theta[j], axes='rzxz'),
                dtype=torch.float,
                device=device
            )
            rotation_global[j] = torch.tensor(
                transforms3d.euler.euler2mat(math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'),
                dtype=torch.float,
                device=device
            )
        # translation[i] = p - distance.unsqueeze(1) * (rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)).squeeze(2)
        # print(f"==>> translation[i]: {translation[i]}")
        translation[i] = p - 0.1 * distance.unsqueeze(1) * (rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)).squeeze(2)
        print(f"==>> translation_: {translation[i]}")
        rotation_hand = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device)
        rotation[i] = rotation_global @ rotation_local @ rotation_hand

    # initialize joint angles
    # joint_angles_mu: hand-crafted canonicalized hand articulation
    # use truncated normal distribution to jitter the joint angles

    joint_angles_mu = torch.tensor([0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0], dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros([n_objects, n_grasps, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, :, i],
            joint_angles_mu[i],
            joint_angles_sigma[i],
            hand_model.joints_lower[i] - 1e-6,
            hand_model.joints_upper[i] + 1e-6
        )
    rotation = T.matrix_to_axis_angle(rotation)
    hand_pose = torch.cat([
        translation,
        rotation,  # axis-angle
        joint_angles,
    ], dim=-1)

    return hand_pose, convex_hulls, torch.stack(ps).cpu().numpy(), torch.stack(closest).cpu().numpy()
    # # initialize contact point indices
    # # contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact], device=device)
    # hand_model.set_parameters(hand_pose)


def init_grasps_test(
    obj_code: str,
    pc_dir: str,
    convex_hull_dir: str,
    scale: float,
    args,
    hand_model,
):
    device = torch.device("cpu")
    n_grasps = 3
    # init
    translation = torch.zeros((n_grasps, 3), dtype=torch.float, device=device)
    rotation = torch.zeros((n_grasps, 3, 3), dtype=torch.float, device=device)

    point_cloud = torch.from_numpy(np.load(osp.join(pc_dir, f"{obj_code}.npy"))).to(device="cuda:7")  # (4096, 3)
    point_cloud *= scale
    enc_xyz = pytorch3d.ops.sample_farthest_points(point_cloud.unsqueeze(0), K=n_grasps)[0]  # (1, n_grasps, 3)
    convex_hull = torch.load(osp.join(convex_hull_dir, f"{obj_code}.pth"), map_location="cuda:7")[scale]  # (1, 409600, 3)
    # convex_hull_down = pytorch3d.ops.sample_farthest_points(convex_hull, K=4096)[0]
    closest_on_hull = pytorch3d.ops.knn_points(enc_xyz, convex_hull, return_nn=True)[2].squeeze().cpu()  # (n_grasps, 3)
    enc_xyz = enc_xyz.squeeze().cpu()  # (n_grasps, 3)
    n = enc_xyz - closest_on_hull
    n /= n.norm(dim=1).unsqueeze(1)  # (n_grasps, 3)
    # convex_hull_np = convex_hull_down.squeeze().cpu().numpy()
    # convex_hull_vis = go.Scatter3d(
    #     x=convex_hull_np[:, 0],
    #     y=convex_hull_np[:, 1],
    #     z=convex_hull_np[:, 2],
    #     marker={"color": "hotpink", "opacity": 0.2},
    # )
    enc_xyz_np = enc_xyz.cpu().numpy()
    enc_xyz_vis = go.Scatter3d(
        x=enc_xyz_np[:, 0],
        y=enc_xyz_np[:, 1],
        z=enc_xyz_np[:, 2],
        marker={"color": "black", "opacity": 0.4},
    )

    distance = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([n_grasps], dtype=torch.float, device=device)
    deviate_theta = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([n_grasps], dtype=torch.float, device=device)
    process_theta = 2 * math.pi * torch.rand([n_grasps], dtype=torch.float, device=device)
    rotate_theta = 2 * math.pi * torch.rand([n_grasps], dtype=torch.float, device=device)

    rotation_local = torch.zeros([n_grasps, 3, 3], dtype=torch.float, device=device)
    rotation_global = torch.zeros([n_grasps, 3, 3], dtype=torch.float, device=device)
    for j in range(n_grasps):
        rotation_local[j] = torch.tensor(
            transforms3d.euler.euler2mat(process_theta[j], deviate_theta[j], rotate_theta[j], axes='rzxz'),
            dtype=torch.float,
            device=device
        )
        rotation_global[j] = torch.tensor(
            transforms3d.euler.euler2mat(math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'),
            dtype=torch.float,
            device=device
        )
    translation = closest_on_hull - 0.2 * distance.unsqueeze(1) * (rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)).squeeze(2)
    rotation_hand = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device).unsqueeze(0)
    rotation = rotation_global @ rotation_local @ rotation_hand

    joint_angles_mu = torch.tensor([0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0], dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros([n_grasps, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i],
            joint_angles_mu[i],
            joint_angles_sigma[i],
            hand_model.joints_lower[i] - 1e-6,
            hand_model.joints_upper[i] + 1e-6
        )
    rotation = T.matrix_to_axis_angle(rotation)
    hand_pose = torch.cat([
        translation,
        rotation,  # axis-angle
        joint_angles,
    ], dim=-1)

    return hand_pose, enc_xyz_vis


if __name__ == "__main__":
    os.chdir("/home/xuguohao/kwokho/DGTR_base")
    from model.utils.hand_model import HandModel

    def parse_args():
        parser = argparse.ArgumentParser()
        # hyper parameters (** Magic, don't touch! **)
        parser.add_argument('--switch_possibility', default=0.5, type=float)
        parser.add_argument('--mu', default=0.98, type=float)
        parser.add_argument('--step_size', default=0.005, type=float)
        parser.add_argument('--stepsize_period', default=50, type=int)
        parser.add_argument('--starting_temperature', default=18, type=float)
        parser.add_argument('--annealing_period', default=30, type=int)
        parser.add_argument('--temperature_decay', default=0.95, type=float)
        parser.add_argument('--w_dis', default=100.0, type=float)
        parser.add_argument('--w_pen', default=100.0, type=float)
        parser.add_argument('--w_spen', default=10.0, type=float)
        parser.add_argument('--w_joints', default=1.0, type=float)
        # initialization settings
        parser.add_argument('--jitter_strength', default=0.1, type=float)
        parser.add_argument('--distance_lower', default=0.2, type=float)
        parser.add_argument('--distance_upper', default=0.3, type=float)
        parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
        parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
        # energy thresholds
        parser.add_argument('--thres_fc', default=0.3, type=float)
        parser.add_argument('--thres_dis', default=0.005, type=float)
        parser.add_argument('--thres_pen', default=0.001, type=float)
        args = parser.parse_args()
        return args

    args = parse_args()
    hand = HandModel(
        mjcf_path="./data/mjcf/shadow_hand.xml",
        mesh_path="./data/mjcf/meshes",
        contact_points_path="./data/mjcf/contact_points.json",
        penetration_points_path="./data/mjcf/penetration_points.json",
        n_surface_points=1024,
        device="cpu",
    )
    mesh_dir = "/home/xuguohao/kwokho/DGTR_base/data/DexGraspNet/meshdata"
    obj_codes = sorted(os.listdir(mesh_dir))
    random.shuffle(obj_codes)
    obj_meshes = []
    NUM_OBJECTS = 5
    available_scales = torch.tensor([0.06, 0.08, 0.1, 0.12, 0.15], dtype=torch.float)
    scales = available_scales[torch.randint(5, (NUM_OBJECTS, ))].numpy().tolist()
    poses = []
    convex_points = []
    for i in range(NUM_OBJECTS):
        code = obj_codes[i]
        obj_meshes.append(tm.load_mesh(osp.join(mesh_dir, code, "coacd", "decomposed.obj")))

        hand_pose, convex = init_grasps_test(
            code,
            "/home/xuguohao/kwokho/DGTR_base/data/DexGraspNet/sampled_meshdata_4096",
            "/home/xuguohao/kwokho/DGTR_base/data/DexGraspNet/sampled_convex_hulls_409600",
            round(scales[i], 2),
            args,
            hand,
        )
        poses.append(hand_pose)
        convex_points.append(convex)

    # poses, convex_hulls, ps, closest = init_grasps(hand, obj_meshes, scales, 5, args)
    # visualize
    for code, i in zip(obj_codes, range(len(obj_meshes))):
        obj_mesh = obj_meshes[i]
        # convex_hull = convex_hulls[i]
        hand_pose = poses[i]
        # p = ps[i]
        # c = closest[i]
        convex = convex_points[i]

        obj_plotly = go.Mesh3d(
            x=obj_mesh.vertices[:, 0] * scales[i], y=obj_mesh.vertices[:, 1] * scales[i], z=obj_mesh.vertices[:, 2] * scales[i],
            i=obj_mesh.faces[:, 0], j=obj_mesh.faces[:, 1], k=obj_mesh.faces[:, 2],
            color="blue",
            opacity=0.9,
        )
        # convex_hull_plotly = go.Mesh3d(
        #     x=convex_hull.vertices[:, 0], y=convex_hull.vertices[:, 1], z=convex_hull.vertices[:, 2],
        #     i=convex_hull.faces[:, 0], j=convex_hull.faces[:, 1], k=convex_hull.faces[:, 2],
        #     color="red",
        #     opacity=.5,
        # )
        hand_plotly = []
        hand.set_parameters(hand_pose)
        for j in range(hand_pose.size(0)):
            hand_plotly.extend(hand.get_plotly_data(j, color="green"))
        # p_plotly = go.Scatter3d(
        #     x=p[:, 0],
        #     y=p[:, 1],
        #     z=p[:, 2],
        #     hovertext="p",
        # )
        # n_plotly = go.Scatter3d(
        #     x=c[:, 0],
        #     y=c[:, 1],
        #     z=c[:, 2],
        #     hovertext="n",
        # )
        # fig = go.Figure([obj_plotly, convex_hull_plotly, p_plotly, n_plotly] + hand_plotly)
        fig = go.Figure([obj_plotly, convex] + hand_plotly)
        fig.update_layout(scene_aspectmode='data')
        fig.add_annotation(text=code, x=0.5, y=0.1, xref='paper', yref='paper')
        save_dir = f"/home/xuguohao/kwokho/DGTR_base/data/vis_init_grasp/init_0.1/{code}"
        os.makedirs(save_dir, exist_ok=True)
        # fig.write_html(osp.join(save_dir, f"{scales[i]}.html"))
        fig.show()
