// Bind the isolated CUDA evidence kernels. Torch owns every device allocation.
#include <ATen/core/Tensor.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <vector>

void evidence_plan(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                   const std::vector<double>&, const std::vector<int64_t>&);
void evidence_connect(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                      const std::vector<double>&);
void evidence_finish(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                     const std::vector<int64_t>&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("plan", &evidence_plan, "Original feature coordinates and corner eligibility");
  module.def("connect", &evidence_connect, "Exact seed-connected component in bounded pixel boxes");
  module.def("finish", &evidence_finish, "Ordered bilinear evidence and original empirical support");
}
