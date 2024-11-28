#ifndef CSDF_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_
#define CSDF_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_

#include <ATen/ATen.h>

namespace csdf {

void unbatched_triangle_distance_forward_cuda(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor normal,
    at::Tensor dist_sign,
    at::Tensor face_idx,
    at::Tensor dist_type);

void unbatched_triangle_distance_backward_cuda(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor face_idx,
    at::Tensor dist_type,
    at::Tensor grad_points);

}  // namespace csdf

#endif // CSDF_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_
