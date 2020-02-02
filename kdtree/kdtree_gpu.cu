#include "kdtree_gpu.hpp"
// clang-format on

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include "../util/cudaCheckError.hpp"

namespace Sculptor {
namespace {
constexpr int kThreads = 128;
constexpr int kBlocks = 16;
constexpr int kStackDepth = 32;
constexpr float kEps = 0.001f;

struct ScaleFunctor {
  __host__ __device__ int operator()(float x) { return scaling_factor * x; }
  __host__ __device__ float operator()(int x) { return descaling_factor * x; }

 private:
  const int scaling_factor = 2048;
  const float descaling_factor = 1.f / scaling_factor;
};
struct EqualToZero {
  __host__ __device__ bool operator()(const int x) { return x == 0; }
};
struct TupleToVec3 {
  __host__ __device__ glm::vec3 operator()(
      thrust::tuple<float, float, float> const& t) {
    return glm::vec3{t.get<0>(), t.get<1>(), t.get<2>()};
  }
};
struct alignas(int) StackEntry {
  int begin;
  int end;
  struct alignas(char) {
    char level;
    char visited_branch;  // left: -1, none: 0, right: 1
  } misc;
};

void ConstructRecursive(thrust::device_vector<int>& x,
                        thrust::device_vector<int>& y,
                        thrust::device_vector<int>& z,
                        int begin,
                        int end) {
  if (end <= begin)
    return;
  auto mid = begin + (end - begin) / 2;
  auto zip =
      thrust::make_zip_iterator(thrust::make_tuple(y.begin(), z.begin()));
  thrust::sort_by_key(thrust::device, x.begin() + begin, x.begin() + end,
                      zip + begin);
  ConstructRecursive(y, z, x, begin, mid);
  ConstructRecursive(y, z, x, mid + 1, end);
}

__host__ __device__ __forceinline__ float dist2(float3 v,
                                                float x,
                                                float y,
                                                float z) {
  return thrust::max(abs(v.x - x), thrust::max(abs(v.y - y), abs(v.z - z)));
}

__global__ void FindToRemoveKernel(float const* const kd_x,
                                   float const* const kd_y,
                                   float const* const kd_z,
                                   int const kd_size,
                                   float const* const query_points,
                                   int* const should_stay,
                                   float const threshold) {
  __shared__ float s_query_pts[3 * kThreads];
  __shared__ StackEntry stack[kStackDepth];
  __shared__ int stack_top;
  __shared__ int go_left_votes;

  {  // retrieve query point
    int tid = 3 * blockIdx.x * blockDim.x + threadIdx.x;
    s_query_pts[threadIdx.x] = query_points[tid];
    s_query_pts[threadIdx.x + blockDim.x] = query_points[tid + blockDim.x];
    s_query_pts[threadIdx.x + 2 * blockDim.x] =
        query_points[tid + 2 * blockDim.x];
    __syncthreads();
  }
  float3 const query_point =
      reinterpret_cast<float3*>(s_query_pts)[threadIdx.x];
  int cur_nearest_point = kd_size / 2;
  float cur_best_dist = dist2(query_point, kd_x[cur_nearest_point],
                              kd_y[cur_nearest_point], kd_z[cur_nearest_point]);
  if (threadIdx.x == 0)
    stack[stack_top = 0] = {0, kd_size, {0, 0}};

FIND_PROC : {
  __syncthreads();
  if (stack[stack_top].end <= stack[stack_top].begin)
    goto RETURN;
  auto mid = (stack[stack_top].begin + stack[stack_top].end) / 2;
  auto dist_to_mid = dist2(query_point, kd_x[mid], kd_y[mid], kd_z[mid]);
  if (dist_to_mid < cur_best_dist) {
    cur_best_dist = dist_to_mid;
    cur_nearest_point = mid;
  }
  if (stack[stack_top].begin + 1 == stack[stack_top].end)
    goto RETURN;
  auto diff = 0.f;
  switch (stack[stack_top].misc.level) {
    default:
    case 0:
      diff = query_point.x - kd_x[mid];
      break;
    case 1:
      diff = query_point.y - kd_y[mid];
      break;
    case 2:
      diff = query_point.z - kd_z[mid];
      break;
  }
  switch (stack[stack_top].misc.visited_branch) {
    default:
    case 0:
      if (threadIdx.x == 0)
        go_left_votes = 0;
      __syncthreads();
      atomicAdd(&go_left_votes, diff < 0);
      __syncthreads();
      if (threadIdx.x == 0) {
        ++stack_top;
        stack[stack_top] = stack[stack_top - 1];
        if (go_left_votes > blockDim.x / 2) {
          stack[stack_top - 1].misc.visited_branch = -1;
          stack[stack_top].end = mid;
        } else {
          stack[stack_top - 1].misc.visited_branch = 1;
          stack[stack_top].begin = mid + 1;
        }
        stack[stack_top].misc.level = stack[stack_top].misc.level == 2
                                          ? 0
                                          : (stack[stack_top].misc.level + 1);
        stack[stack_top].misc.visited_branch = 0;
      }
      goto FIND_PROC;
    case 1:
      if (__syncthreads_or(diff < kEps)) {
        if (threadIdx.x == 0) {
          stack[stack_top].end = mid;
          stack[stack_top].misc.level = stack[stack_top].misc.level == 2
                                            ? 0
                                            : (stack[stack_top].misc.level + 1);
          stack[stack_top].misc.visited_branch = 0;
        }
        goto FIND_PROC;
      }
      goto RETURN;
    case -1:
      if (__syncthreads_or(diff > -kEps)) {
        if (threadIdx.x == 0) {
          stack[stack_top].begin = mid + 1;
          stack[stack_top].misc.level = stack[stack_top].misc.level == 2
                                            ? 0
                                            : (stack[stack_top].misc.level + 1);
          stack[stack_top].misc.visited_branch = 0;
        }
        goto FIND_PROC;
      }
      goto RETURN;
  }
}
RETURN : {
  if (stack_top > 0) {
    __syncthreads();
    if (threadIdx.x == 0)
      --stack_top;
    goto FIND_PROC;
  }
}
  if (cur_best_dist < threshold)
    should_stay[cur_nearest_point] = 0;
}
}  // namespace

void KdTreeGPU::Construct(float* x, float* y, float* z, int size) {
  thrust::device_vector<int> x_int(size), y_int(size), z_int(size);

  thrust::transform(x, x + size, x_int.begin(), ScaleFunctor());
  thrust::transform(y, y + size, y_int.begin(), ScaleFunctor());
  thrust::transform(z, z + size, z_int.begin(), ScaleFunctor());

  ConstructRecursive(x_int, y_int, z_int, 0, size);

  thrust::transform(x_int.begin(), x_int.end(), x, ScaleFunctor());
  thrust::transform(y_int.begin(), y_int.end(), y, ScaleFunctor());
  thrust::transform(z_int.begin(), z_int.end(), z, ScaleFunctor());
}

std::vector<glm::vec3> KdTreeGPU::RemoveNearest(float* x,
                                                float* y,
                                                float* z,
                                                int kd_size,
                                                float* query_points,
                                                int query_points_size,
                                                float threshold) {
  int* dshould_stay;
  SculptorCudaCheckError(cudaMalloc(reinterpret_cast<void**>(&dshould_stay),
                                    sizeof(int) * kd_size));
  thrust::fill(thrust::device_ptr<int>(dshould_stay),
               thrust::device_ptr<int>(dshould_stay) + kd_size, 1);

  int iteration = 0;
  int excess = query_points_size % (kThreads * kBlocks);
  int max = query_points_size - excess;
  std::vector<cudaStream_t> streams;
  streams.reserve((max / (kThreads * kBlocks) + 2));
  for (; iteration < max; iteration += kThreads * kBlocks) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    FindToRemoveKernel<<<kBlocks, kThreads, 0, stream>>>(
        x, y, z, kd_size, query_points + 3 * iteration, dshould_stay,
        threshold);
    streams.emplace_back(stream);
  }
  if (excess >= kThreads) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    FindToRemoveKernel<<<excess / kThreads, kThreads, 0, stream>>>(
        x, y, z, kd_size, query_points + 3 * iteration, dshould_stay,
        threshold);
    streams.emplace_back(stream);
  }
  iteration += (excess / kThreads) * kThreads;
  excess %= kThreads;
  if (excess > 0) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    FindToRemoveKernel<<<1, excess, 0, stream>>>(x, y, z, kd_size,
                                                 query_points + 3 * iteration,
                                                 dshould_stay, threshold);
    streams.emplace_back(stream);
  }
  for (auto const& stream : streams) {
    SculptorCudaCheckError(cudaStreamSynchronize(stream));
    SculptorCudaCheckError(cudaStreamDestroy(stream));
  }
  streams.clear();

  thrust::device_ptr<int> should_stay_dev_ptr(dshould_stay);
  auto xyz = thrust::make_zip_iterator(thrust::make_tuple(
      thrust::device_ptr<float>(x), thrust::device_ptr<float>(y),
      thrust::device_ptr<float>(z)));
  auto to_remove = kd_size - thrust::reduce(should_stay_dev_ptr,
                                            should_stay_dev_ptr + kd_size);
  thrust::device_vector<thrust::tuple<float, float, float>> ret_dev_raw(
      to_remove);
  auto ret_dev_raw_end =
      thrust::copy_if(xyz, xyz + kd_size, should_stay_dev_ptr,
                      ret_dev_raw.begin(), EqualToZero());
  thrust::device_vector<glm::vec3> ret_dev(ret_dev_raw_end -
                                           ret_dev_raw.begin());
  thrust::transform(ret_dev_raw.begin(), ret_dev_raw_end, ret_dev.begin(),
                    TupleToVec3());
  std::vector<glm::vec3> ret(ret_dev_raw_end - ret_dev_raw.begin());
  thrust::copy(ret_dev.begin(), ret_dev.end(), ret.begin());

  thrust::remove_if(xyz, xyz + kd_size, should_stay_dev_ptr, EqualToZero());
  SculptorCudaCheckError(cudaFree(dshould_stay));

  return ret;
}
}  // namespace Sculptor
