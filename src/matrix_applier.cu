
#include "../include/matrix_applier.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"

namespace MatrixApplier {
namespace {
constexpr int kMatrixSize = 4;
constexpr int kThreads = 128;
constexpr int kBlocks = 16;

__constant__ float c_matrix[kMatrixSize][kMatrixSize];

void __global__ ApplyKernel(float* vectors, int size) {
  __shared__ float s_data[3 * kThreads];
  for (int index = 3 * blockIdx.x * kThreads + threadIdx.x; index < size;
       index += kBlocks * kThreads * 3) {
    __syncthreads();
    s_data[threadIdx.x] = vectors[index];
    if (index + kThreads < size)
      s_data[threadIdx.x + kThreads] = vectors[index + kThreads];
    if (index + 2 * kThreads < size)
      s_data[threadIdx.x + 2 * kThreads] = vectors[index + 2 * kThreads];
    __syncthreads();
    float3 v = reinterpret_cast<float3*>(s_data)[threadIdx.x];
    reinterpret_cast<float3*>(s_data)[threadIdx.x] = {
        v.x * c_matrix[0][0] + v.y * c_matrix[1][0] + v.z * c_matrix[2][0] +
            c_matrix[3][0],
        v.x * c_matrix[0][1] + v.y * c_matrix[1][1] + v.z * c_matrix[2][1] +
            c_matrix[3][1],
        v.x * c_matrix[0][2] + v.y * c_matrix[1][2] + v.z * c_matrix[2][2] +
            c_matrix[3][2]};
    __syncthreads();
    vectors[index] = s_data[threadIdx.x];
    if (index + kThreads < size)
      vectors[index + kThreads] = s_data[threadIdx.x + kThreads];
    if (index + 2 * kThreads < size)
      vectors[index + 2 * kThreads] = s_data[threadIdx.x + 2 * kThreads];
  }
}  // namespace
}  // namespace

void Apply(glm::vec3* vectors, int n, glm::mat4 matrix) {
  float* d_vectors;
  cudaMalloc(&d_vectors, 3 * sizeof(float) * n);
  cudaMemcpy(d_vectors, vectors, 3 * sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0);
  ApplyKernel<<<kBlocks, kThreads>>>(d_vectors, 3 * n);
  cudaDeviceSynchronize();
  cudaMemcpy(vectors, d_vectors, 3 * sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaFree(d_vectors);
}
}  // namespace MatrixApplier
