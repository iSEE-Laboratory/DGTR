import trimesh
import torch
import csdf
import os

os.environ["CUDA_VISIBLE_DIVICES"] = "1"

device = "cuda"

test_mesh = trimesh.primitives.Sphere(radius=1)
samples, _ = trimesh.sample.sample_surface(test_mesh, 1000)
verts = torch.Tensor(test_mesh.vertices.copy()).to(device)
faces = torch.Tensor(test_mesh.faces.copy()).long().to(device)
face_verts = csdf.index_vertices_by_faces(verts, faces)

x = torch.Tensor(samples).to(device).requires_grad_()
x = x*1.01

print("points:", x.shape)
print("face_verts:", face_verts.shape)

dis, normal, dis_sign, dis_faces, dis_types = csdf.compute_sdf(x, face_verts)
print("squared distance:", dis.shape)
print("normal:", normal.shape)
gradient_by_normal = 2 * dis.unsqueeze(1).sqrt() * normal
gradient = torch.autograd.grad([dis.sum()], [x], create_graph=True,
                               retain_graph=True)[0]
print("check normal:", torch.allclose(gradient, gradient_by_normal, atol=2e-7))
