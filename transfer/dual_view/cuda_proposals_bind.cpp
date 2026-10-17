// Bind the local finite-growth kernel to Torch-owned CUDA memory and stream.
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/pybind.h>

void launch_growth(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, const at::Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("grow", &launch_growth,
             "Grow fixed CPU random plans, pausing ambiguous KDTree boundary rows");
}
