#include <torch/extension.h>

#include "unbatched_triangle_distance.h"

namespace csdf {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unbatched_triangle_distance_forward_cuda",
              &unbatched_triangle_distance_forward_cuda);
  m.def("unbatched_triangle_distance_backward_cuda",
              &unbatched_triangle_distance_backward_cuda);
}

}  // namespace csdf
