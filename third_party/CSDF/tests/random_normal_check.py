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

pass_cnt = 0
total = 0

for i in np.random.choice(len(codes_all), test_num):
    code = codes_all[i]
    print(code)
    files = os.listdir(os.path.join(root_path, code, "coacd"))
    meshes = trimesh.load(os.path.join(
        root_path, code, "coacd/decomposed.obj"), force="mesh", process=False).split()
    for mesh in meshes:
        total += 2
        samples, _ = trimesh.sample.sample_surface(mesh, 10000)
        verts = torch.Tensor(mesh.vertices.copy()).to(device)
        faces = torch.Tensor(mesh.faces.copy()).long().to(device)
        x = torch.Tensor(samples).to(device).requires_grad_()
        center = verts.mean(0, keepdim=True)
        x = 1.1*(x-center) + center
        face_verts = index_vertices_by_faces(verts, faces)
        dis, normal, dis_sign, dis_faces, dis_types = csdf.compute_sdf(
            x, face_verts)
        normal_f = normal * 2 * dis.unsqueeze(1).sqrt()
        normal_g = torch.autograd.grad([dis.sum()], [x], create_graph=True,
                                       retain_graph=True)[0]
        if torch.allclose(normal_f, normal_g, atol=2e-7):
            pass_cnt += 1
        # print(torch.allclose(normal_f, normal_g, atol=2e-7))
        x = 0.9*(x-center) + center
        dis, normal, dis_sign, dis_faces, dis_types = csdf.compute_sdf(
            x, face_verts)
        normal_f = normal * 2 * dis.unsqueeze(1).sqrt()
        normal_g = torch.autograd.grad([dis.sum()], [x], create_graph=True,
                                       retain_graph=True)[0]
        if torch.allclose(normal_f, normal_g, atol=2e-7):
            pass_cnt += 1
        # print(torch.allclose(normal_f, normal_g, atol=2e-7))
print(pass_cnt, "/", total)
