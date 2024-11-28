# CSDF

This is my custom signed distance(SDF) computation of points to manifold mesh with PyTorch API on GPU. The code and framework is modified from [Kaolin](https://github.com/NVIDIAGameWorks/kaolin).

## Installation

```bash
python setup.py install
```

If encounter `circular import`, try:

```bash
pip install -e .
```

or manually modify `csdf/__init__.py`.

## Usage

The code provide two function:

- `compute_sdf(pointclouds, face_vertices)`
    - input
        - unbatched points with shape (num, 3)
        - unbatched face_vertices with shape (num , 3, 3)
    - returns 
        - squared distance
        - normal defined by gradient
        - distance signs (inside -1 and outside 1)
        - closest face indexes
        - distance type (plane, vertices or edges)
- `index_vertices_by_faces(vertices_features, faces)`: return face_verts reqired by `compute_sdf(pointclouds, face_vertices)`.

## Note

- Sign is defined by `sign(dis.dot(face_normal))`, **check your mesh has perfect normal information**.
- Returned normal is defined by `(p - closest_point).normalized()` or equally $\frac{\partial d}{\partial p}$, not face normal.
- The code only run on cuda.
- Scripts in `tests` cannnot run independently (requires kaolin api or mesh datasets).
