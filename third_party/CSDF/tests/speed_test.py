import trimesh
import kaolin
from csdf import index_vertices_by_faces
import torch
import os
import csdf
import time

os.environ["CUDA_VISIBLE_DIVICES"] = "1"

device = "cuda"

for mesh in os.listdir("meshes"):
    test_mesh = trimesh.load(os.path.join(
        "meshes", mesh), force="mesh", process=False)
    samples, _ = trimesh.sample.sample_surface(test_mesh, 1000000)
    verts = torch.Tensor(test_mesh.vertices.copy()).to(device).unsqueeze(0)
    verts_ = torch.Tensor(test_mesh.vertices.copy()).to(device)
    faces = torch.Tensor(test_mesh.faces.copy()).long().to(device)
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(verts, faces)
    face_verts_ = index_vertices_by_faces(verts_, faces)

    t1 = time.time()
    for i in range(100):
        x = torch.Tensor(samples).to(device).requires_grad_()
        x = x*1.01
        dis, dis_faces, dis_types = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            x.unsqueeze(0), face_verts)
        signs_ = kaolin.ops.mesh.check_sign(verts, faces, x.unsqueeze(0))
        g = torch.autograd.grad([dis.sum()], [x], create_graph=True,
                                retain_graph=True)[0]
    t2 = time.time()
    print(f"kaolin: {t2-t1}s")

    t1 = time.time()
    for i in range(100):
        x = torch.Tensor(samples).to(device).requires_grad_()
        x = x*1.01
        dis_, normal, dis_sign, dis_faces, dis_types = csdf.compute_sdf(
            x, face_verts_)
    t2 = time.time()
    print(f"csdf: {t2-t1}s")
    break
    