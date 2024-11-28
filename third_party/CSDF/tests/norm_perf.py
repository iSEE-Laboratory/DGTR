import trimesh
import numpy as np
import os

data_root = "data"

cats = os.listdir(data_root)
for cat in cats:
    codes = os.listdir(os.path.join(data_root, cat))
    for code in codes:
        mesh_path = os.path.join(data_root, cat, code, "coacd/decomposed.obj")
        print(mesh_path)
        meshes = trimesh.load(mesh_path, force="mesh", process=False).split()
        cnt = 0
        real_cnt = 0
        for mesh in meshes:
            verts = mesh.vertices
            center = np.mean(verts, 0)
            real_cnt += mesh.faces.shape[0]
            for face in mesh.faces:
                v1 = verts[face[0], :]
                v2 = verts[face[1], :]
                v3 = verts[face[2], :]
                normal = np.cross(v1-v2, v1-v3)
                line = (v1+v2+v3)/3 - center
                if np.dot(normal, line) >= 0:
                    cnt += 1
        print("count:", cnt)
        print("total:", real_cnt)
        if(cnt != real_cnt):
            print("dif", real_cnt-cnt)
