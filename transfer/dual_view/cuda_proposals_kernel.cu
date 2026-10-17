// One block per fixed growth: sequential source rows, parallel ranked targets.
// Every scalar operation explicitly rounds binary64 to nearest. In particular,
// no FMA, TF32, fast sqrt, or parallel floating reduction changes NumPy's
// three-coordinate squared norm ((dx*dx + dy*dy) + dz*dz).
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
// BEGIN explicitly declare CUDA's double infinity constant for nextafter.
#include <math_constants.h>
// END do not depend on incidental transitive CUDA-header inclusion.
#include <climits>
#include <cstdint>

namespace {
constexpr int THREADS = 128;

__device__ __forceinline__ double squared_distance(const double* a, const double* b) {
  const double x = __dsub_rn(a[0], b[0]);
  const double y = __dsub_rn(a[1], b[1]);
  const double z = __dsub_rn(a[2], b[2]);
  return __dadd_rn(__dadd_rn(__dmul_rn(x, x), __dmul_rn(y, y)), __dmul_rn(z, z));
}

__global__ void grow_kernel(const double* xyz, const double* distances,
    const int* ranked, const int64_t* offsets, const int* prefixes,
    const int* permutations, const double* scales, const double* bands,
    const double* outer_squared, const double* uncertainty, int* all_state,
    int64_t* work, int query_count) {
  const int group = blockIdx.x, lane = threadIdx.x;
  int* state = all_state+group*28;
  if (state[2] == 2) return;
  __shared__ int selected_rows[12], selected_columns[12];
  __shared__ int count, position, status;
  __shared__ int minimum_rank[THREADS];
  __shared__ int64_t totals[THREADS];
  if (lane < 12) {
    selected_rows[lane] = state[4+lane];
    selected_columns[lane] = state[16+lane];
  }
  if (lane == 0) { count = state[0]; position = state[1]; status = 0; }
  __syncthreads();
  int64_t rank_work = 0, distance_work = 0, row_work = 0;
  const double scale = scales[group], band = bands[group];
  const int permutation_count = query_count-2;
  while (position < permutation_count && count < 12) {
    const int row = permutations[group*permutation_count+position];
    const int prefix = prefixes[group*query_count+row];
    if (lane == 0 && prefix) ++row_work;
    bool row_finished = false;
    for (int64_t begin = 0; begin < prefix; begin += THREADS) {
      const int64_t rank = begin+lane;
      int winner = INT_MAX;
      if (rank < prefix) {
        ++rank_work;
        const int column = ranked[offsets[row]+rank];
        bool compatible = true;
        for (int anchor = 0; anchor < count; ++anchor) {
          if (selected_columns[anchor] == column) { compatible = false; break; }
        }
        for (int anchor = 0; compatible && anchor < count; ++anchor) {
          const double squared = squared_distance(xyz+3*column, xyz+3*selected_columns[anchor]);
          const double actual = __dsqrt_rn(squared);
          const double desired = __dmul_rn(scale,
              distances[static_cast<int64_t>(row)*query_count+selected_rows[anchor]]);
          ++distance_work;
          compatible = fabs(__dsub_rn(actual, desired)) <= band;
        }
        if (compatible) winner = static_cast<int>(rank);
      }
      minimum_rank[lane] = winner;
      __syncthreads();
      for (int stride = THREADS/2; stride; stride /= 2) {
        if (lane < stride) minimum_rank[lane] = min(minimum_rank[lane], minimum_rank[lane+stride]);
        __syncthreads();
      }
      if (minimum_rank[0] != INT_MAX) {
        if (lane == 0) {
          const int column = ranked[offsets[row]+minimum_rank[0]];
          const double squared = squared_distance(xyz+3*column, xyz+3*selected_columns[0]);
          const double outer = outer_squared[group*query_count+row];
          // Do not discard any point using a device approximation to the CPU
          // tree. Pause before appending the smallest compatible rank if tree
          // membership can affect that decision. The CPU certifies this row.
          const double definitely_inside = nextafter(__dsub_rn(outer, uncertainty[group]),
                                                     -CUDART_INF);
          if (!isfinite(squared) || !isfinite(outer) || squared >= definitely_inside) {
            status = 1;
          } else {
            selected_rows[count] = row;
            selected_columns[count] = column;
            ++count;
          }
        }
        __syncthreads();
        row_finished = true;
      }
      if (row_finished) break;
    }
    if (status == 1) break;
    if (lane == 0) ++position;
    __syncthreads();
  }
  if (lane < 12) {
    state[4+lane] = selected_rows[lane];
    state[16+lane] = selected_columns[lane];
  }
  if (lane == 0) {
    state[0] = count;
    state[1] = position;
    state[2] = status == 1 ? 1 : 2;
  }
  // Integer work accounting is independent of selection and is accumulated
  // across boundary resumes; all floating outputs remain the original CPU fit.
  const int64_t local[] = {rank_work, distance_work, row_work};
  for (int counter = 0; counter < 3; ++counter) {
    totals[lane] = local[counter];
    __syncthreads();
    for (int stride = THREADS/2; stride; stride /= 2) {
      if (lane < stride) totals[lane] += totals[lane+stride];
      __syncthreads();
    }
    if (lane == 0) work[group*3+counter] += totals[0];
    __syncthreads();
  }
}
}  // namespace

void launch_growth(const at::Tensor& xyz, const at::Tensor& distances,
    const at::Tensor& ranked, const at::Tensor& offsets, const at::Tensor& prefixes,
    const at::Tensor& permutations, const at::Tensor& scales, const at::Tensor& bands,
    const at::Tensor& outer_squared, const at::Tensor& uncertainty,
    const at::Tensor& state, const at::Tensor& work) {
  for (const auto& tensor : {xyz, distances, ranked, offsets, prefixes, permutations,
                            scales, bands, outer_squared, uncertainty, state, work}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.device() == xyz.device(),
                "Proposal kernel inputs must be contiguous tensors on one CUDA device");
    TORCH_CHECK(!tensor.requires_grad(), "Proposal growth is a forward-only integer selection");
  }
  for (const auto& tensor : {xyz, distances, scales, bands, outer_squared, uncertainty})
    TORCH_CHECK(tensor.scalar_type() == at::kDouble, "Proposal geometry must be FP64");
  for (const auto& tensor : {ranked, prefixes, permutations, state})
    TORCH_CHECK(tensor.scalar_type() == at::kInt, "Proposal indices/state must be int32");
  TORCH_CHECK(offsets.scalar_type() == at::kLong && work.scalar_type() == at::kLong,
              "Proposal offsets and work counters must be int64");
  const int64_t groups = state.size(0), queries = distances.size(0);
  TORCH_CHECK(queries >= 6 && queries < INT_MAX && groups > 0 && groups <= 128,
              "Proposal kernel expects the reviewed finite nonempty plan");
  TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3 && xyz.size(0) < INT_MAX &&
              distances.dim() == 2 && distances.size(1) == queries &&
              offsets.numel() == queries+1 && ranked.dim() == 1 &&
              prefixes.sizes() == at::IntArrayRef({groups, queries}) &&
              permutations.sizes() == at::IntArrayRef({groups, queries-2}) &&
              scales.numel() == groups && bands.numel() == groups &&
              outer_squared.sizes() == at::IntArrayRef({groups, queries}) &&
              uncertainty.numel() == groups &&
              state.sizes() == at::IntArrayRef({groups, 28}) &&
              work.sizes() == at::IntArrayRef({groups, 3}), "Invalid proposal tensor shapes");
  const c10::cuda::CUDAGuard guard(xyz.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(xyz.get_device());
  grow_kernel<<<groups, THREADS, 0, stream.stream()>>>(xyz.data_ptr<double>(),
      distances.data_ptr<double>(), ranked.data_ptr<int>(), offsets.data_ptr<int64_t>(),
      prefixes.data_ptr<int>(), permutations.data_ptr<int>(), scales.data_ptr<double>(),
      bands.data_ptr<double>(), outer_squared.data_ptr<double>(), uncertainty.data_ptr<double>(),
      state.data_ptr<int>(), work.data_ptr<int64_t>(), static_cast<int>(queries));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
