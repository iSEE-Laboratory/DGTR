from concurrent.futures import process
from rsa import sign
import trimesh
import kaolin
from csdf import index_vertices_by_faces
import torch
import os
import csdf
import numpy as np

os.environ["CUDA_VISIBLE_DIVICES"] = "1"

device = "cuda"

root_path = "data"

codes_all = []

for cat in os.listdir(root_path):
    codes = os.listdir(os.path.join(root_path, cat))
    codes_all += [cat+"/"+code for code in codes]


test_num = 100

min_ratio =1.

for i in np.random.choice(len(codes_all), test_num):
    code = codes_all[i]
    print(code)
    files = os.listdir(os.path.join(root_path, code, "coacd"))
    meshes = trimesh.load(os.path.join(
        root_path, code, "coacd/decomposed.obj"), force="mesh", process=False).split()
    for mesh in meshes:
        samples, _ = trimesh.sample.sample_surface(mesh, 10000)
        verts = torch.Tensor(mesh.vertices.copy()).to(device)
        faces = torch.Tensor(mesh.faces.copy()).long().to(device)
        x = torch.Tensor(samples).to(device).requires_grad_()
        center = verts.mean(0, keepdim=True)
        x = 1.1*(x-center) + center
        face_verts = index_vertices_by_faces(verts, faces)
        dis, _, dis_sign, dis_faces, dis_types = csdf.compute_sdf(x, face_verts)
        cnt = int((dis_sign > 0).sum())
        if((cnt/dis_sign.shape[0]) < min_ratio):
            min_ratio = (cnt/dis_sign.shape[0])
        # print("outside:", cnt, "/", dis_sign.shape[0])
        x = 0.9*(x-center) + center
        dis, _, dis_sign, dis_faces, dis_types = csdf.compute_sdf(x, face_verts)
        cnt = int((dis_sign < 0).sum())
        if((cnt/dis_sign.shape[0]) < min_ratio):
            min_ratio = (cnt/dis_sign.shape[0])
        # print("inside:", cnt, "/", dis_sign.shape[0])
print(min_ratio)