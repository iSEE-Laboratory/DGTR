import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import random
import shutil
from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import plotly.graph_objects as go
import pytorch3d.ops
import pytorch3d.structures
import pytorch3d.transforms as T
import pytorch_kinematics as pk
import torch
import trimesh as tm
from csdf import compute_sdf, index_vertices_by_faces
from tqdm import tqdm, trange


class HandModelWithPlot:
    def __init__(self, mjcf_path, mesh_path, contact_points_path, penetration_points_path, n_surface_points=0, device="cpu"):
        """
        Create a Hand Model for a MJCF robot

        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        mesh_path: str
            path to mesh directory
        contact_points_path: str
            path to hand-selected contact candidates
        penetration_points_path: str
            path to hand-selected penetration keypoints
        n_surface_points: int
            number of points to sample from surface of hand, use fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device

        # load articulation

        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        # load contact points and penetration points

        contact_points = json.load(open(contact_points_path, "r")) if contact_points_path is not None else None
        penetration_points = json.load(open(penetration_points_path, "r")) if penetration_points_path is not None else None

        # build mesh

        self.mesh = {}
        areas = {}

        def build_mesh_recurse(body):
            if (len(body.link.visuals) > 0):
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "box":
                        # link_mesh = trimesh.primitives.Box(extents=2 * visual.geom_param)
                        link_mesh = tm.load_mesh(os.path.join(mesh_path, "box.obj"), process=False)
                        link_mesh.vertices *= visual.geom_param.detach().cpu().numpy()
                    elif visual.geom_type == "capsule":
                        link_mesh = tm.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "mesh":
                        link_mesh = tm.load_mesh(os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".obj"), process=False)
                        if visual.geom_param[1] is not None:
                            scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(self.device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                contact_candidates = torch.tensor(contact_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if contact_points is not None else None
                penetration_keypoints = torch.tensor(penetration_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if penetration_points is not None else None
                self.mesh[link_name] = {
                    "vertices": link_vertices,
                    "faces": link_faces,
                    "contact_candidates": contact_candidates,
                    "penetration_keypoints": penetration_keypoints,
                }
                if link_name in ["robot0:palm", "robot0:palm_child", "robot0:lfmetacarpal_child"]:
                    link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                    self.mesh[link_name]["face_verts"] = link_face_verts
                else:
                    self.mesh[link_name]["geom_param"] = body.link.visuals[0].geom_param
                areas[link_name] = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
            for children in body.children:
                build_mesh_recurse(children)
        build_mesh_recurse(self.chain._root)

        # set joint limits

        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

        # sample surface points

        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]["surface_points"] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]["vertices"].unsqueeze(0), self.mesh[link_name]["faces"].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]["surface_points"] = surface_points

        # indexing

        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))

        self.contact_candidates = [self.mesh[link_name]["contact_candidates"] for link_name in self.mesh]
        self.global_index_to_link_index = sum([[i] * len(contact_candidates) for i, contact_candidates in enumerate(self.contact_candidates)], [])
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(self.global_index_to_link_index, dtype=torch.long, device=device)
        self.n_contact_candidates = self.contact_candidates.shape[0]

        self.penetration_keypoints = [self.mesh[link_name]["penetration_keypoints"] for link_name in self.mesh]
        self.global_index_to_link_index_penetration = sum([[i] * len(penetration_keypoints) for i, penetration_keypoints in enumerate(self.penetration_keypoints)], [])
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(self.global_index_to_link_index_penetration, dtype=torch.long, device=device)
        self.n_keypoints = self.penetration_keypoints.shape[0]

        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

    def set_parameters(self, hand_pose, contact_point_indices=None):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+3+`n_dofs`) torch.FloatTensor
            translation, rotation in axisangle, and joint angles
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = T.axis_angle_to_matrix(self.hand_pose[:, 3:6])
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 6:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device)
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, n_contact, 4, 4)
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat([self.contact_points, torch.ones(batch_size, n_contact, 1, dtype=torch.float, device=self.device)], dim=2)
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[:, :, :3, 0]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)

    def get_plotly_data(self, i, opacity=0.5, color="lightblue", with_contact_points=False, pose=None):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]["vertices"])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=opacity))
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2], mode="markers", marker=dict(color="red", size=5)))
        return data


class Visualizer:
    def __init__(
        self,
        shuffle,
        obj_mesh_root,
        mjcf_path,
        mesh_path,
        contact_points_path,
        penetration_points_path
    ) -> None:
        self.obj_mesh_root = obj_mesh_root
        self.hand = HandModelWithPlot(
            mjcf_path=mjcf_path,
            mesh_path=mesh_path,
            contact_points_path=contact_points_path,
            penetration_points_path=penetration_points_path,
        )
        self.save_path = save_path
        self.shuffle = shuffle
        self.device = torch.device("cpu")

    def visualize_result(
        self,
        results: List,
        num_objs: Union[int, str] = "all",
        num_grasps: Union[int, str] = "all",
    ):
        """
        Params:
            results: A list of test results
            num_objs: Number of objects to visualize
            num_grasps: Number of grasps for each objects to visualize
        """
        if self.shuffle:
            random.shuffle(results)
        num_objs = num_objs if isinstance(num_objs, int) else len(results)

        pbar = tqdm(total=num_objs, desc="Visualizing", ncols=120)
        def update_pbar(_): return pbar.update(1)
        pool = mp.Pool(16)
        for i in range(num_objs):
            data = results[i]
            data["predictions"] = torch.tensor(data["predictions"])
            data["targets"] = torch.tensor(data["targets"])
            pool.apply_async(
                self._visualize_one_result,
                args=(data, num_grasps),
                callback=update_pbar
            )
        pool.close()
        pool.join()

    def _visualize_one_result(self, data: Dict, num_grasps: int):
        # visualize object
        obj_code = data["obj_code"]
        scale = data["scale"]
        obj_save_path = osp.join(self.save_path, obj_code, str(scale))
        os.makedirs(obj_save_path, exist_ok=True)
        obj_mesh = tm.load_mesh(osp.join(self.obj_mesh_root, obj_code, "coacd", "decomposed.obj"))
        vertices = obj_mesh.vertices
        faces = obj_mesh.faces
        vertices = vertices * scale
        obj_plotly = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=1,
        )
        # visualize gt hand
        gt_poses = data["targets"]
        pred_poses = data["predictions"]
        if self.shuffle:
            shuffle_idx = torch.randperm(len(gt_poses)).numpy().tolist()
        else:
            shuffle_idx = torch.arange(len(gt_poses), dtype=torch.long).numpy().tolist()
        num_grasps = num_grasps if isinstance(num_grasps, int) else len(gt_poses)
        for i in range(num_grasps):
            idx = shuffle_idx[i]
            pred_pose = pred_poses[idx]
            gt_pose = gt_poses[idx]

            pred_hand = deepcopy(self.hand)
            pred_hand.set_parameters(pred_pose.unsqueeze(0))
            pred_hand_plotly = pred_hand.get_plotly_data(i=0, opacity=1, color="lightblue")

            gt_hand = deepcopy(self.hand)
            gt_hand.set_parameters(gt_pose.unsqueeze(0))
            gt_hand_plotly = gt_hand.get_plotly_data(i=0, opacity=.5, color="red")

            fig = go.Figure([obj_plotly] + pred_hand_plotly + gt_hand_plotly)
            fig.write_html(
                file=osp.join(osp.join(obj_save_path, f"pred_{i}.html")),
                include_plotlyjs="directory",
                full_html=True,
            )


if __name__ == "__main__":
    """
    Usage: python ./visualize_results.py -r ./results.json -s -o 40
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path", type=str, required=True, help="path for results.json saved by test.py")
    parser.add_argument("-s", "--shuffle", action="store_true", help="whether to shuffle objects")
    parser.add_argument("-o", "--num_objs", type=int, default=-1, help="number of objects to visualize (-1 means all)")
    parser.add_argument("-g", "--num_grasps", type=int, default=-1, help="number of grasps to visualize for each object (-1 means all)")
    args = parser.parse_args()

    num_objs = args.num_objs if args.num_objs != -1 else "all"
    num_grasps = args.num_grasps if args.num_grasps != -1 else all
    result_path = args.result_path
    save_path = osp.join(osp.dirname(osp.abspath(result_path)), "visualization")
    try:
        os.makedirs(save_path)
    except FileExistsError:
        print(f"{save_path} already exists.")
        input("Press Enter to EMPTY it and recreate a NEW one, or Ctrl-C to exit")
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    visualizer = Visualizer(
        shuffle=args.shuffle,
        obj_mesh_root="/home/xuguohao/kwokho/DGTR_base/data/DexGraspNet/meshdata",
        mjcf_path="/home/xuguohao/kwokho/DGTR_base/data/mjcf/shadow_hand_wrist_free.xml",
        mesh_path="/home/xuguohao/kwokho/DGTR_base/data/mjcf/meshes",
        contact_points_path="/home/xuguohao/kwokho/DGTR_base/data/mjcf/contact_points.json",
        penetration_points_path="/home/xuguohao/kwokho/DGTR_base/data/mjcf/penetration_points.json",
    )
    with open(result_path, "r") as rf:
        results = json.load(rf)
    visualizer.visualize_result(results, num_objs=num_objs, num_grasps=num_grasps)
