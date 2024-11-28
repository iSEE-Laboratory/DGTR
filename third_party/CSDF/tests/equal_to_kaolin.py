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
    verts = torch.Tensor(test_mesh.vertices.copy()).to(device).unsqueeze(0)
    verts_ = torch.Tensor(test_mesh.vertices.copy()).to(device)
    faces = torch.Tensor(test_mesh.faces.copy()).long().to(device)
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(verts, faces)
    face_verts_ = index_vertices_by_faces(verts_, faces)
    if not torch.allclose(face_verts, face_verts_.unsqueeze(0)):
        print("wrong")
        exit(0)

    x = torch.Tensor(samples).to(device).requires_grad_()
    x = x*1.01
    dis, dis_faces, dis_types = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        x.unsqueeze(0), face_verts)
    signs_ = kaolin.ops.mesh.check_sign(verts, faces, x.unsqueeze(0))
    signs = torch.where(signs_, -torch.ones_like(
        signs_).int(), torch.ones_like(signs_).int())
    g = torch.autograd.grad([dis.sum()], [x], create_graph=True,
                            retain_graph=True)[0]

    dis_, normal, dis_sign, dis_faces, dis_types = csdf.compute_sdf(
        x, face_verts_)
    if mesh != "suzanne.obj":
        if not torch.allclose(signs, dis_sign):
            print("wrong in sign")
            exit(0)
    g__ = 2 * dis_.unsqueeze(1).sqrt() * normal
    g_ = torch.autograd.grad([dis_.sum()], [x], create_graph=True,
                             retain_graph=True)[0]
    if not torch.allclose(g_, g__, atol=2e-7):
        print("bug in normal")
    if not torch.allclose(g, g_, atol=2e-7):
        # for i in range(g.shape[0]):
        #     if not torch.allclose(g[i], g_[i]):
        #         print(g[i])
        #         print(g_[i])
        #         print(g_[i]-g[i])
        print("wrong in normal")
        exit(0)
    if not torch.allclose(dis, dis_):
        print("wrong in dis")
        exit(0)

    x = torch.Tensor(samples).to(device).requires_grad_()
    x = x/1.01
    dis, dis_faces, dis_types = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        x.unsqueeze(0), face_verts)
    signs_ = kaolin.ops.mesh.check_sign(verts, faces, x.unsqueeze(0))
    signs = torch.where(signs_, -torch.ones_like(
        signs_).int(), torch.ones_like(signs_).int())
    g = torch.autograd.grad([dis.sum()], [x], create_graph=True,
                            retain_graph=True)[0]

    dis_, normal, dis_sign, dis_faces, dis_types = csdf.compute_sdf(
        x, face_verts_)
    g__ = 2 * dis_.unsqueeze(1).sqrt() * normal
    g_ = torch.autograd.grad([dis_.sum()], [x], create_graph=True,
                             retain_graph=True)[0]
    if not torch.allclose(g_, g__, atol=2e-7):
        print("bug in normal")
    if mesh != "suzanne.obj":
        if not torch.allclose(signs, dis_sign):
            print("wrong in sign")
            exit(0)
    if not torch.allclose(g, g_, atol=2e-7):
        # for i in range(g.shape[0]):
        #     if not torch.allclose(g[i], g_[i]):
        #         print(g[i])
        #         print(g_[i])
        #         print(g_[i]-g[i])
        print("wrong in normal")
        exit(0)
    if not torch.allclose(dis, dis_):
        print("wrong in dis")
        exit(0)

print("pass test")
