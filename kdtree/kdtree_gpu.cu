#include "kdtree_gpu.hpp"
// clang-format on

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <future>

namespace Sculptor {
namespace {
constexpr int kThreads = 128;
constexpr int kBlocks = 32;
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
  int begin;
  int end;
  struct alignas(char) {
    char level;
    char visited_branch;  // left: -1, none: 0, right: 1
  } misc;
};

__global__ void FindToRemoveKernel(float const* kd_x,
                                   float const* kd_y,
                                   float const* kd_z,
                                   int const kd_size,
                                   float const* query_points,
                                   int const number_of_query_points,
                                   int* to_remove,
                                   int* to_remove_top,
                                   float const threshold) {
  __shared__ float s_query_pts[3 * kThreads];
  __shared__ StackEntry stack[kStackDepth];
  __shared__ int stack_top;
  __shared__ int go_left, go_right;

  // retrieve query point
  int tid = 3 * blockIdx.x * blockDim.x + threadIdx.x;
  s_query_pts[threadIdx.x] = query_points[tid];
  s_query_pts[threadIdx.x + kThreads] = query_points[tid + kThreads];
  s_query_pts[threadIdx.x + 2 * kThreads] = query_points[tid + 2 * kThreads];
  __syncthreads();
  float3 query_point = reinterpret_cast<float3*>(s_query_pts)[threadIdx.x];

  // pre-procedure tasks
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0)
    stack[stack_top = 0] = {0, kd_size, {0, 0}};
  int cur_nearest_point = kd_size / 2;
  float cur_best_dist = dist2(query_point, kd_x[cur_nearest_point],
                              kd_y[cur_nearest_point], kd_z[cur_nearest_point]);
PROC_START : {  // procedure find (begin, end, level)
  __syncthreads();
  if (stack[stack_top].end <= stack[stack_top].begin)
    goto RETURN;  // no node to investigate
  // calculate root node
  auto mid = (stack[stack_top].end + stack[stack_top].begin) / 2;
  auto dist_to_mid = dist2(query_point, kd_x[mid], kd_y[mid], kd_z[mid]);
  // update current best node if necessary
  if (dist_to_mid < cur_best_dist) {
    cur_best_dist = dist_to_mid;
    cur_best_dist = mid;
  }
  if (stack[stack_top].begin + 1 == stack[stack_top].end)
    goto RETURN;  // no subtrees to visit

  auto diff = 0.f;
  switch (stack[stack_top].misc.level) {
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
    case 0:
      break;
    case -1:
      // vote whether we need to visit right subtree:
      if (__syncthreads_or(diff <= -kEps && -diff < cur_best_dist)) {
        // replace current stack frame as we wont be going back to this node
        if (threadIdx.x) {
          stack[stack_top].begin = mid + 1;
          stack[stack_top].misc.level = stack[stack_top].misc.level == 2
                                            ? 0
                                            : (stack[stack_top].misc.level + 1);
          // mark we havent visited any sub tree of right subtree
          stack[stack_top].misc.visited_branch = 0;
        }
        goto PROC_START;
      } else {
        // proceed to removing current stack frame as we dont need to visit
        // right subtree
        goto RETURN;
      }
    case 1:
      // vote whether we need to visit left subtree
      if (__syncthreads_or(diff >= kEps && diff < cur_best_dist)) {
        // replace current stack frame as we wont be going back to this node
        if (threadIdx.x == 0) {
          stack[stack_top].end = mid;
          stack[stack_top].misc.level = stack[stack_top].misc.level == 2
                                            ? 0
                                            : (stack[stack_top].misc.level + 1);
          // mark we havent visited any sub tree of right subtree
          stack[stack_top].misc.visited_branch = 0;
        }
        goto PROC_START;
      } else {
        goto RETURN;
      }
  }
  // here we havent visited any of current node subtrees. threads will vote
  // which subtree they wanna visit first. whichever wins will be visited.
  if (threadIdx.x == 0)
    go_left = go_right = 0;
  __syncthreads();
  // voting time
  atomicAdd(&go_left, diff < kEps);
  atomicAdd(&go_right, diff > -kEps);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (go_left > go_right) {
      // record this choice in current stack frame
      stack[stack_top].misc.visited_branch = -1;
      // create next stack frame
      ++stack_top;
      stack[stack_top].begin = stack[stack_top - 1].begin;
      stack[stack_top].end = mid;
    } else {
      // record this choice in current stack frame
      stack[stack_top].misc.visited_branch = 1;
      // create next stack frame
      ++stack_top;
      stack[stack_top].begin = stack[stack_top - 1].begin;
      stack[stack_top].end = mid;
    }
    stack[stack_top].misc.level = stack[stack_top - 1].misc.level == 2
                                      ? 0
                                      : (stack[stack_top - 1].misc.level + 1);
  }
  goto PROC_START;
}
RETURN : {
  if (stack_top == 0) {
    if (cur_best_dist < threshold) {
      auto i = atomicAdd(to_remove_top, 1);
      to_remove[i] = cur_nearest_point;
    }
    return;
  }
  if (threadIdx.x == 0)
    --stack_top;
  goto PROC_START;
}
}
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

  int to_remove_size = 0;
  thrust::device_vector<int> nearest_neighbour_index(query_points.GetSize(),
                                                     -1);
  FindToRemoveKernel<<<kBlocks, kThreads>>>(
      dx, dy, dz, x.GetSize(), dq, query_points.GetSize(),
      thrust::raw_pointer_cast(nearest_neighbour_index.data()), &to_remove_size,
      threshold);
  cudaDeviceSynchronize();

  cudaGraphicsUnmapResources(1, &query_res);
  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);
}
}  // namespace Sculptor
