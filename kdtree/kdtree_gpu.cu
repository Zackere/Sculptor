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
  return fmaxf(fabsf(v.x - x), fmaxf(fabsf(v.y - y), fabsf(v.z - z)));
}

struct alignas(int) StackEntry {
  int begin = 0;
  int end = 0;
  struct alignas(char) {
    char level = 0;
    char visited_branch = 0;  // left: -1, none: 0, right: 1
  } misc;
};

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
    s_query_pts[threadIdx.x + kThreads] = query_points[tid + kThreads];
    s_query_pts[threadIdx.x + 2 * kThreads] = query_points[tid + 2 * kThreads];
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
      break;
    case 1:
      if (__syncthreads_or(diff < kEps && -diff < cur_best_dist)) {
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
      if (__syncthreads_or(diff > -kEps && diff < cur_best_dist)) {
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

  if (threadIdx.x == 0)
    go_left_votes = 0;
  __syncthreads();
  atomicAdd(&go_left_votes, diff < 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    ++stack_top;
    stack[stack_top] = stack[stack_top - 1];
    if (go_left_votes > kThreads / 2) {
      stack[stack_top - 1].misc.visited_branch = -1;
      stack[stack_top].end = mid;
    } else {
      stack[stack_top - 1].misc.visited_branch = 1;
      stack[stack_top].begin = mid + 1;
    }
    stack[stack_top].misc.level = stack[stack_top - 1].misc.level == 2
                                      ? 0
                                      : (stack[stack_top - 1].misc.level + 1);
    stack[stack_top].misc.visited_branch = 0;
  }
  goto FIND_PROC;
}
RETURN : {
  if (stack_top > 0) {
    if (threadIdx.x == 0)
      --stack_top;
    goto FIND_PROC;
  }
}
  if (cur_best_dist < threshold)
    should_stay[cur_nearest_point] = 0;
}

struct EqualToZero {
  __host__ __device__ bool operator()(const int x) { return x == 0; }
};

}  // namespace

void KdTreeGPU::Construct(CudaGraphicsResource<float>& x,
                          CudaGraphicsResource<float>& y,
                          CudaGraphicsResource<float>& z) {
  auto *x_res = x.GetCudaResource(), *y_res = y.GetCudaResource(),
       *z_res = z.GetCudaResource();

  cudaGraphicsMapResources(1, &x_res);
  cudaGraphicsMapResources(1, &y_res);
  cudaGraphicsMapResources(1, &z_res);

  thrust::device_ptr<float> dx, dy, dz;
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dx),
                                       &num_bytes, x_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dy),
                                       &num_bytes, y_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dz),
                                       &num_bytes, z_res);

  thrust::device_vector<int> x_int(x.GetSize()), y_int(y.GetSize()),
      z_int(z.GetSize());

  thrust::transform(dx, dx + x.GetSize(), x_int.begin(), ScaleFunctor());
  thrust::transform(dy, dy + y.GetSize(), y_int.begin(), ScaleFunctor());
  thrust::transform(dz, dz + z.GetSize(), z_int.begin(), ScaleFunctor());

  ConstructRecursive(x_int, y_int, z_int, 0, x_int.size());

  thrust::transform(x_int.begin(), x_int.end(), dx, ScaleFunctor());
  thrust::transform(y_int.begin(), y_int.end(), dy, ScaleFunctor());
  thrust::transform(z_int.begin(), z_int.end(), dz, ScaleFunctor());

  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);
}

void KdTreeGPU::RemoveNearest(CudaGraphicsResource<float>& x,
                              CudaGraphicsResource<float>& y,
                              CudaGraphicsResource<float>& z,
                              CudaGraphicsResource<glm::vec3>& query_points,
                              float threshold,
                              bool construct) {
  auto *x_res = x.GetCudaResource(), *y_res = y.GetCudaResource(),
       *z_res = z.GetCudaResource();
  auto* query_res = query_points.GetCudaResource();
  cudaGraphicsMapResources(1, &x_res);
  cudaGraphicsMapResources(1, &y_res);
  cudaGraphicsMapResources(1, &z_res);
  cudaGraphicsMapResources(1, &query_res);

  float *dx, *dy, *dz;
  float* dq;
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dx),
                                       &num_bytes, x_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dy),
                                       &num_bytes, y_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dz),
                                       &num_bytes, z_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dq),
                                       &num_bytes, query_res);

  int* dshould_stay;
  cudaMalloc(reinterpret_cast<void**>(&dshould_stay),
             sizeof(int) * x.GetSize());
  thrust::fill(thrust::device_ptr<int>(dshould_stay),
               thrust::device_ptr<int>(dshould_stay) + x.GetSize(), 1);

  int iteration = 0;
  int max = query_points.GetSize() - kThreads * kBlocks;
  for (; iteration < max; iteration += kThreads * kBlocks)
    FindToRemoveKernel<<<kBlocks, kThreads>>>(
        dx, dy, dz, x.GetSize(), dq + 3 * iteration, dshould_stay, threshold);
  FindToRemoveKernel<<<(query_points.GetSize() - iteration) / kThreads,
                       kThreads>>>(dx, dy, dz, x.GetSize(), dq + 3 * iteration,
                                   dshould_stay, threshold);
  cudaDeviceSynchronize();
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
  thrust::device_ptr<int> should_stay_dev_ptr(dshould_stay);
  auto xyz = thrust::make_zip_iterator(thrust::make_tuple(
      thrust::device_ptr<float>(dx), thrust::device_ptr<float>(dy),
      thrust::device_ptr<float>(dz)));
  auto removed = xyz + x.GetSize() -
                 thrust::remove_if(xyz, xyz + x.GetSize(), should_stay_dev_ptr,
                                   EqualToZero());
  if (removed) {
    x.PopBack(removed);
    y.PopBack(removed);
    z.PopBack(removed);
  }
  cudaFree(dshould_stay);

  cudaGraphicsUnmapResources(1, &query_res);
  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);
}
}  // namespace Sculptor
