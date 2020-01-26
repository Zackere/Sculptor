#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "matrix_applier.hpp"

namespace Sculptor {
namespace {
constexpr int kMatrixSize = 4;
constexpr int kThreads = 128;
constexpr int kBlocks = 32;

__constant__ float c_matrix[kMatrixSize][kMatrixSize];

void __global__ ApplyUnrestrictedKernel(float* vectors, int size) {
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
}  // namespace
void MatrixApplier::Apply(std::vector<glm::vec3>& vectors,
                          glm::mat4 const& matrix) {
  auto extra_space = vectors.size() % (kThreads * kBlocks);
  if (extra_space != 0)
    extra_space = kThreads * kBlocks - extra_space;

  float* dvectors = nullptr;
  cudaMalloc(&dvectors, sizeof(float) * 3 * (vectors.size() + extra_space));
  cudaMemcpy(dvectors, vectors.data(), sizeof(float) * 3 * vectors.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0);
  ApplyUnrestrictedKernel<<<kBlocks, kThreads>>>(
      dvectors, 3 * (vectors.size() + extra_space));
  cudaMemcpy(vectors.data(), dvectors, sizeof(float) * 3 * vectors.size(),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(dvectors);
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

  ApplyUnrestrictedKernel<<<kBlocks, kThreads>>>(dvectors, 3 * nvectors);
  cudaDeviceSynchronize();

  cudaGraphicsUnmapResources(1, &vectors);
}
}  // namespace Sculptor
