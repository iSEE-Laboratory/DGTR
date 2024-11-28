from torch.autograd import gradcheck
import trimesh
import kaolin
from csdf import index_vertices_by_faces
import torch
import os
import csdf

os.environ["CUDA_VISIBLE_DIVICES"] = "1"

device = "cuda"

for mesh in os.listdir("meshes"):
    test_mesh = trimesh.load(os.path.join(
        "meshes", mesh), force="mesh", process=False)
    samples, _ = trimesh.sample.sample_surface(test_mesh, 1000)
    verts = torch.Tensor(test_mesh.vertices.copy()).to(device)
    faces = torch.Tensor(test_mesh.faces.copy()).long().to(device)
    face_verts = index_vertices_by_faces(verts, faces).double()

    x = torch.Tensor(samples).to(device).requires_grad_().double()
    x = x*1.01
    inputs = (x, face_verts)
    test = gradcheck(csdf.compute_sdf, inputs)
    print(test)
    x = torch.Tensor(samples).to(device).requires_grad_().double()
    x = x/1.01
    inputs = (x, face_verts)
    test = gradcheck(csdf.compute_sdf, inputs)
    print(test)
