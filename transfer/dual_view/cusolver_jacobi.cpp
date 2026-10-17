// Binding to NVIDIA's existing batched Jacobi solver for dual-view geometry.
// Only FP64 CUDA [B,3,3] forward inputs are supported. Torch owns all tensors,
// the current device/stream, the borrowed solver handle and workspace allocation.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/pybind.h>
#include <limits>
#include <tuple>

namespace {
void check_solver(cusolverStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS,
              operation, " failed with cuSOLVER status ", static_cast<int>(status));
}

// BEGIN exception-safe ownership of only the Jacobi parameter object.
struct JacobiParameters {
  syevjInfo_t value = nullptr;
  JacobiParameters() {
    check_solver(cusolverDnCreateSyevjInfo(&value), "cusolverDnCreateSyevjInfo");
  }
  JacobiParameters(const JacobiParameters&) = delete;
  JacobiParameters& operator=(const JacobiParameters&) = delete;
  void close() {
    const auto owned = value;
    value = nullptr;
    check_solver(cusolverDnDestroySyevjInfo(owned), "cusolverDnDestroySyevjInfo");
  }
  ~JacobiParameters() noexcept {
    if (value != nullptr) {
      const auto status = cusolverDnDestroySyevjInfo(value);
      // Never replace an in-flight exception with a throwing destructor.
      if (status != CUSOLVER_STATUS_SUCCESS) {
        TORCH_WARN("Jacobi parameter cleanup failed with cuSOLVER status ",
                   static_cast<int>(status));
      }
    }
  }
};
// END the Torch solver handle is borrowed and is never destroyed here.

std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t>
eigh3x3(const at::Tensor& input, bool query_only) {
  TORCH_CHECK(input.is_cuda(), "Jacobi input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kDouble, "Jacobi input must be FP64");
  TORCH_CHECK(input.dim() == 3 && input.size(1) == 3 && input.size(2) == 3,
              "Jacobi input must have shape [B,3,3]");
  TORCH_CHECK(!input.requires_grad(), "This forward geometry binding has no autograd rule");
  TORCH_CHECK(input.size(0) <= std::numeric_limits<int>::max(), "Batch exceeds cuSOLVER int range");
  const int batch = static_cast<int>(input.size(0));
  const c10::cuda::CUDAGuard guard(input.device());
  auto values = at::empty({batch, 3}, input.options());
  // Explicit column-major matrices preserve the logical LOWER triangle even
  // when the caller's unused upper triangle contains unrelated values.
  auto vectors = at::empty_strided({batch, 3, 3}, {9, 1, 3}, input.options());
  auto info = at::empty({batch}, input.options().dtype(at::kInt));
  if (batch == 0) {
    return {values, vectors, info.cpu(), 0};
  }
  vectors.copy_(input);
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  const auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  check_solver(cusolverDnSetStream(handle, stream.stream()), "cusolverDnSetStream");
  JacobiParameters parameters;
  // Retain library defaults: machine-accuracy tolerance and 100 sweeps.
  check_solver(cusolverDnXsyevjSetSortEig(parameters.value, 1), "cusolverDnXsyevjSetSortEig");
  int lwork = 0;
  check_solver(cusolverDnDsyevjBatched_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER, 3, vectors.data_ptr<double>(), 3,
      values.data_ptr<double>(), &lwork, parameters.value, batch),
      "cusolverDnDsyevjBatched_bufferSize");
  TORCH_CHECK(lwork >= 0, "cuSOLVER returned a negative workspace size");
  // A separately exposed size query returns no numerical outputs. This lets
  // the driver record/validate the required allocation before a large solve.
  if (query_only) {
    parameters.close();
    return {at::Tensor(), at::Tensor(), at::Tensor(), static_cast<int64_t>(lwork)};
  }
  // lwork counts double elements, not bytes. Torch's allocator enforces the
  // process memory fraction; no hidden cudaMalloc allocation is introduced.
  auto workspace = at::empty({static_cast<int64_t>(lwork)}, input.options());
  check_solver(cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER, 3, vectors.data_ptr<double>(), 3,
      values.data_ptr<double>(), workspace.data_ptr<double>(), lwork,
      info.data_ptr<int>(), parameters.value, batch), "cusolverDnDsyevjBatched");
  // This explicit D2H is part of the call's time. It synchronizes the current
  // stream and checks every matrix; nonconvergence never invokes CPU fallback.
  auto host_info = info.cpu();
  const int* status = host_info.const_data_ptr<int>();
  int failures = 0, first_index = -1, first_status = 0;
  for (int index = 0; index < batch; ++index) {
    if (status[index] != 0) {
      if (failures == 0) { first_index = index; first_status = status[index]; }
      ++failures;
    }
  }
  parameters.close();
  TORCH_CHECK(failures == 0, "cusolverDnDsyevjBatched returned ", failures,
              " nonzero info entries; first matrix=", first_index, ", info=", first_status);
  return {values, vectors, host_info, static_cast<int64_t>(lwork)};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("eigh3x3", [](const at::Tensor& input) { return eigh3x3(input, false); },
             "FP64 CUDA lower-triangle batched Jacobi: W,V,checked CPU info,lwork doubles",
             py::arg("input"));
  module.def("workspace_lwork", [](const at::Tensor& input) {
    return std::get<3>(eigh3x3(input, true));
  }, "Query Jacobi workspace in double elements without running the solver", py::arg("input"));
}
