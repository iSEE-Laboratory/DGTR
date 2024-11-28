#include <ATen/ATen.h>

#include "check.h"

namespace csdf {

#ifdef WITH_CUDA

void unbatched_triangle_distance_forward_cuda_impl(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor normal,
    at::Tensor dist_sign,
    at::Tensor face_idx,
    at::Tensor dist_type);

void unbatched_triangle_distance_backward_cuda_impl(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor face_idx,
    at::Tensor dist_type,
    at::Tensor grad_points);

#endif  // WITH_CUDA


void unbatched_triangle_distance_forward_cuda(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor normal,
    at::Tensor dist_sign,
    at::Tensor face_idx,
    at::Tensor dist_type) {
  CHECK_CUDA(points);
  CHECK_CUDA(face_vertices);
  CHECK_CUDA(dist);
  CHECK_CUDA(normal);
  CHECK_CUDA(dist_sign);
  CHECK_CUDA(face_idx);
  CHECK_CUDA(dist_type);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(face_vertices);
  CHECK_CONTIGUOUS(dist);
  CHECK_CONTIGUOUS(normal);
  CHECK_CONTIGUOUS(dist_sign);
  CHECK_CONTIGUOUS(face_idx);
  CHECK_CONTIGUOUS(dist_type);
  const int num_points = points.size(0);
  const int num_faces = face_vertices.size(0);
  CHECK_SIZES(points, num_points, 3);
  CHECK_SIZES(face_vertices, num_faces, 3, 3);
  CHECK_SIZES(normal, num_points, 3);
  CHECK_SIZES(dist, num_points);
  CHECK_SIZES(dist_sign, num_points);
  CHECK_SIZES(face_idx, num_points);
  CHECK_SIZES(dist_type, num_points);
#if WITH_CUDA
  unbatched_triangle_distance_forward_cuda_impl(
      points, face_vertices, dist, normal, dist_sign, face_idx, dist_type);
#else
  AT_ERROR("unbatched_triangle_distance not built with CUDA");
#endif
}

void unbatched_triangle_distance_backward_cuda(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor face_idx,
    at::Tensor dist_type,
    at::Tensor grad_points) {
  CHECK_CUDA(grad_dist);
  CHECK_CUDA(points);
  CHECK_CUDA(face_vertices);
  CHECK_CUDA(face_idx);
  CHECK_CUDA(dist_type);
  CHECK_CUDA(grad_points);
  CHECK_CONTIGUOUS(grad_dist);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(face_vertices);
  CHECK_CONTIGUOUS(face_idx);
  CHECK_CONTIGUOUS(dist_type);
  CHECK_CONTIGUOUS(grad_points);

  const int num_points = points.size(0);
  const int num_faces = face_vertices.size(0);
  CHECK_SIZES(grad_dist, num_points);
  CHECK_SIZES(points, num_points, 3);
  CHECK_SIZES(face_vertices, num_faces, 3, 3);
  CHECK_SIZES(face_idx, num_points);
  CHECK_SIZES(dist_type, num_points);
  CHECK_SIZES(grad_points, num_points, 3);

#if WITH_CUDA
  unbatched_triangle_distance_backward_cuda_impl(
      grad_dist, points, face_vertices, face_idx, dist_type,
      grad_points);
#else
  AT_ERROR("unbatched_triangle_distance_backward not built with CUDA");
#endif
}

}  // namespace csdf
