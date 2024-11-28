import torch
from csdf import _C


def index_vertices_by_faces(vertices_features, faces):
    r"""Index vertex features to convert per vertex tensor to per vertex per face tensor.

    Args:
        vertices_features (torch.FloatTensor):
            vertices features, of shape
            :math:`(\text{batch_size}, \text{num_points}, \text{knum})`,
            ``knum`` is feature dimension, the features could be xyz position,
            rgb color, or even neural network features.
        faces (torch.LongTensor):
            face index, of shape :math:`(\text{num_faces}, \text{num_vertices})`.
    Returns:
        (torch.FloatTensor):
            the face features, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{num_vertices}, \text{knum})`.
    """
    assert vertices_features.ndim == 2, \
        "vertices_features must have 2 dimensions of shape (num_points, knum)"
    assert faces.ndim == 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)"
    input = vertices_features.reshape(
        -1, 1, 3).expand(-1, faces.shape[-1], -1)
    indices = faces[..., None].expand(
        -1, -1, vertices_features.shape[-1])
    return torch.gather(input=input, index=indices, dim=0)


def compute_sdf(pointclouds, face_vertices):
    return _UnbatchedTriangleDistanceCuda.apply(pointclouds, face_vertices)


class _UnbatchedTriangleDistanceCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, face_vertices):
        assert not face_vertices.requires_grad, "not support face_vertices gradient"
        num_points = points.shape[0]
        num_faces = face_vertices.shape[0]
        min_dist = torch.zeros(
            (num_points), device=points.device, dtype=points.dtype)
        normal = torch.zeros(
            (num_points, 3), device=points.device, dtype=points.dtype)
        dist_sign = torch.zeros(
            (num_points), device=points.device, dtype=torch.int32)
        min_dist_idx = torch.zeros(
            (num_points), device=points.device, dtype=torch.long)
        dist_type = torch.zeros(
            (num_points), device=points.device, dtype=torch.int32)
        _C.unbatched_triangle_distance_forward_cuda(
            points, face_vertices, min_dist, normal, dist_sign, min_dist_idx, dist_type)
        ctx.save_for_backward(points.contiguous(), face_vertices.contiguous(),
                              min_dist_idx, dist_type)
        ctx.mark_non_differentiable(normal, dist_sign, min_dist_idx, dist_type)
        return min_dist, normal, dist_sign, min_dist_idx, dist_type

    @staticmethod
    def backward(ctx, grad_dist, grad_normal, grad_dist_sign, grad_face_idx, grad_dist_type):
        points, face_vertices, face_idx, dist_type = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_points = torch.zeros_like(points)
        grad_face_vertices = None
        _C.unbatched_triangle_distance_backward_cuda(
            grad_dist, points, face_vertices, face_idx, dist_type,
            grad_points)
        return grad_points, grad_face_vertices
