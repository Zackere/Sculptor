
#include "../include/matrix_applier.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "thrust/device_vector.h"

namespace MatrixApplier {
namespace {
constexpr int kMatrixSize = 4;
constexpr int kThreads = 256;
constexpr int kBlocks = 16;

__constant__ float c_matrix[kMatrixSize][kMatrixSize];

void __global__ ApplyKernel(float* vectors, int size) {
  __shared__ float s_data[3 * kThreads];
  for (int index = 3 * kThreads * blockIdx.x + threadIdx.x; index < size;
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
}
}  // namespace

void Apply(std::vector<glm::vec3>& vectors, glm::mat4 matrix) {
  thrust::device_vector<glm::vec3> d_vectors;
  auto extra_space = vectors.size() % (kThreads * kBlocks);
  if (extra_space != 0)
    extra_space = kThreads * kBlocks - extra_space;
  d_vectors.resize(vectors.size() + extra_space);
  thrust::copy(vectors.begin(), vectors.end(), d_vectors.begin());
  cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0);
  ApplyKernel<<<kBlocks, kThreads>>>(
      reinterpret_cast<float*>(d_vectors.data().get()), 3 * d_vectors.size());
  cudaDeviceSynchronize();
  d_vectors.resize(vectors.size());
  thrust::copy(d_vectors.begin(), d_vectors.end(), vectors.begin());
}
}  // namespace MatrixApplier
