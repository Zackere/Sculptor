#include "matrix_applier.hpp"
// clang-format on

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>

namespace Sculptor {
namespace {
constexpr int kMatrixSize = 4;
constexpr int kThreads = 128;
constexpr int kBlocks = 32;

__constant__ float c_matrix[kMatrixSize][kMatrixSize];

__global__ void ApplyKernel(float* vectors, int size) {
  __shared__ float s_data[3 * kThreads];
  int index = 0;
  int off = 3 * kThreads * kBlocks - threadIdx.x - 1;
  size -= off;
  for (index = 3 * kThreads * blockIdx.x + threadIdx.x; index < size;
       index += 3 * kThreads * kBlocks) {
    s_data[threadIdx.x] = vectors[index];
    s_data[threadIdx.x + kThreads] = vectors[index + kThreads];
    s_data[threadIdx.x + 2 * kThreads] = vectors[index + 2 * kThreads];
    __syncthreads();
    float3 v = reinterpret_cast<float3*>(s_data)[threadIdx.x];
    reinterpret_cast<float3*>(s_data)[threadIdx.x] =
        float3{v.x * c_matrix[0][0] + v.y * c_matrix[1][0] +
                   v.z * c_matrix[2][0] + c_matrix[3][0],
               v.x * c_matrix[0][1] + v.y * c_matrix[1][1] +
                   v.z * c_matrix[2][1] + c_matrix[3][1],
               v.x * c_matrix[0][2] + v.y * c_matrix[1][2] +
                   v.z * c_matrix[2][2] + c_matrix[3][2]};
    __syncthreads();
    vectors[index] = s_data[threadIdx.x];
    vectors[index + kThreads] = s_data[threadIdx.x + kThreads];
    vectors[index + 2 * kThreads] = s_data[threadIdx.x + 2 * kThreads];
  }
  size += off;
  __syncthreads();
  // Handle leftover vectors
  if (index < size)
    s_data[threadIdx.x] = vectors[index];
  if (index + kThreads < size)
    s_data[threadIdx.x + kThreads] = vectors[index + kThreads];
  if (index + 2 * kThreads < size)
    s_data[threadIdx.x + 2 * kThreads] = vectors[index + 2 * kThreads];
  __syncthreads();
  float3 v = reinterpret_cast<float3*>(s_data)[threadIdx.x];
  reinterpret_cast<float3*>(s_data)[threadIdx.x] =
      float3{v.x * c_matrix[0][0] + v.y * c_matrix[1][0] +
                 v.z * c_matrix[2][0] + c_matrix[3][0],
             v.x * c_matrix[0][1] + v.y * c_matrix[1][1] +
                 v.z * c_matrix[2][1] + c_matrix[3][1],
             v.x * c_matrix[0][2] + v.y * c_matrix[1][2] +
                 v.z * c_matrix[2][2] + c_matrix[3][2]};
  __syncthreads();
  if (index < size)
    vectors[index] = s_data[threadIdx.x];
  if (index + kThreads < size)
    vectors[index + kThreads] = s_data[threadIdx.x + kThreads];
  if (index + 2 * kThreads < size)
    vectors[index + 2 * kThreads] = s_data[threadIdx.x + 2 * kThreads];
}
__global__ void ApplyKernel(float* x, float* y, float* z, int size) {
  for (int i = kThreads * blockIdx.x + threadIdx.x; i < size;
       i += kThreads * kBlocks) {
    float3 v{x[i], y[i], z[i]};
    v = float3{v.x * c_matrix[0][0] + v.y * c_matrix[1][0] +
                   v.z * c_matrix[2][0] + c_matrix[3][0],
               v.x * c_matrix[0][1] + v.y * c_matrix[1][1] +
                   v.z * c_matrix[2][1] + c_matrix[3][1],
               v.x * c_matrix[0][2] + v.y * c_matrix[1][2] +
                   v.z * c_matrix[2][2] + c_matrix[3][2]};
    x[i] = v.x;
    y[i] = v.y;
    z[i] = v.z;
  }
}
}  // namespace
void MatrixApplier::Apply(std::vector<glm::vec3>& vectors,
                          glm::mat4 const& matrix) {
  thrust::device_vector<glm::vec3> dvectors = vectors;
  cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0);
  ApplyKernel<<<kBlocks, kThreads>>>(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(dvectors.data())),
      3 * dvectors.size());
  cudaDeviceSynchronize();
  thrust::copy(dvectors.begin(), dvectors.end(), vectors.begin());
}

void MatrixApplier::Apply(cudaGraphicsResource* vectors,
                          int nvectors,
                          glm::mat4 const& matrix) {
  cudaGraphicsMapResources(1, &vectors);
  float* dvectors = nullptr;
  size_t num_bytes;
  cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dvectors),
                                       &num_bytes, vectors);

  ApplyKernel<<<kBlocks, kThreads>>>(dvectors, 3 * nvectors);
  cudaDeviceSynchronize();

  cudaGraphicsUnmapResources(1, &vectors);
}
void MatrixApplier::Apply(cudaGraphicsResource* x,
                          cudaGraphicsResource* y,
                          cudaGraphicsResource* z,
                          int nvectors,
                          glm::mat4 const& matrix) {
  cudaGraphicsMapResources(1, &x);
  cudaGraphicsMapResources(1, &y);
  cudaGraphicsMapResources(1, &z);

  float *dx, *dy, *dz;
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dx),
                                       &num_bytes, x);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dy),
                                       &num_bytes, y);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dz),
                                       &num_bytes, z);
  cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0);
  ApplyKernel<<<kBlocks, kThreads>>>(dx, dy, dz, nvectors);
  cudaDeviceSynchronize();

  cudaGraphicsUnmapResources(1, &z);
  cudaGraphicsUnmapResources(1, &y);
  cudaGraphicsUnmapResources(1, &x);
}

std::unique_ptr<MatrixApplierBase> MatrixApplier::Clone() const {
  return std::make_unique<MatrixApplier>();
}
}  // namespace Sculptor
