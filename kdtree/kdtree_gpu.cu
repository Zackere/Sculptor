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

struct ScaleFunctor {
  __host__ __device__ int operator()(float x) { return x * scaling_factor; }
  __host__ __device__ float operator()(int x) {
    return static_cast<float>(x) / scaling_factor;
  }

 private:
  const int scaling_factor = 2048;
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

__host__ __device__ __forceinline__ float dist2(float3 v1,
                                                float x,
                                                float y,
                                                float z) {
  return (v1.x - x) * (v1.x - x) + (v1.y - y) * (v1.y - y) +
         (v1.z - z) * (v1.z - z);
}

__global__ void FindNearestKernel(float const* kd_x,
                                  float const* kd_y,
                                  float const* kd_z,
                                  int const kd_size,
                                  float const* query_points,
                                  int const number_of_query_points,
                                  int* nearest_neighbour_index) {
  __shared__ float s_query_pts[3 * kThreads];
  __shared__ int3 stack[kStackDepth];
  __shared__ int stack_top;

  int tid = 3 * blockIdx.x * blockDim.x + threadIdx.x;
  s_query_pts[threadIdx.x] = query_points[tid];
  s_query_pts[threadIdx.x + kThreads] = query_points[tid + kThreads];
  s_query_pts[threadIdx.x + 2 * kThreads] = query_points[tid + 2 * kThreads];
  __syncthreads();
  float3 query_point = reinterpret_cast<float3*>(s_query_pts)[threadIdx.x];

  tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0)
    stack[stack_top = 0] = {0, kd_size, 0};
  int cur_nearest_point = kd_size / 2;
  float cur_best_dist = dist2(query_point, kd_x[cur_nearest_point],
                              kd_y[cur_nearest_point], kd_z[cur_nearest_point]);
  __syncthreads();
PROC_START : {  // procedure find (begin, end, level)
  if (stack[stack_top].y <= stack[stack_top].x)
    goto RETURN;
  // calculate root node
  auto mid = (stack[stack_top].y - stack[stack_top].x) / 2;
  float dist_to_mid = dist2(query_point, kd_x[mid], kd_y[mid], kd_z[mid]);
  // update current best node
  if (dist_to_mid < cur_best_dist) {
    cur_best_dist = dist_to_mid;
    cur_best_dist = mid;
  }
}
RETURN : {
  if (stack_top == 0) {
    nearest_neighbour_index[tid] = cur_nearest_point;
    return;
  }
  if (threadIdx.x == 0)
    --stack_top;
  __syncthreads();
  goto PROC_START;
}
  printf("BYE");
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

  ConstructRecursive(x_int, y_int, z_int, 0, x.GetSize());

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
                              float threshold) {
  threshold = threshold;
  thrust::device_vector<int> nearest_neighbour_index(query_points.GetSize(),
                                                     -1);

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

  FindNearestKernel<<<kBlocks, kThreads>>>(
      dx, dy, dz, x.GetSize(), dq, query_points.GetSize(),
      thrust::raw_pointer_cast(nearest_neighbour_index.data()));

  cudaGraphicsUnmapResources(1, &query_res);
  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);
}
}  // namespace Sculptor
