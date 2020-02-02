#include "matrix_applier.hpp"
// clang-format on

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>

#include <utility>

#include "../util/cudaCheckError.hpp"

namespace Sculptor {
namespace {
constexpr int kMatrixSize = 4;
constexpr int kThreads = 256;
constexpr int kBlocks = 64;

__constant__ float c_matrix[kMatrixSize][kMatrixSize];

__global__ void ApplyKernel(float* vectors, int offset) {
  __shared__ float s_data[3 * kThreads];
  int index = 3 * blockDim.x * blockIdx.x + threadIdx.x + offset;
  s_data[threadIdx.x] = vectors[index];
  s_data[threadIdx.x + blockDim.x] = vectors[index + blockDim.x];
  s_data[threadIdx.x + 2 * blockDim.x] = vectors[index + 2 * blockDim.x];
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
  vectors[index + blockDim.x] = s_data[threadIdx.x + blockDim.x];
  vectors[index + 2 * blockDim.x] = s_data[threadIdx.x + 2 * blockDim.x];
}
__global__ void ApplyKernel(float* x, float* y, float* z, int offset) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
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

template <typename OffsetCalculator, typename... Args>
void Launch(int datasize, OffsetCalculator offset_calculator, Args... args) {
  int excess = datasize % (kThreads * kBlocks);
  int max = datasize - excess;
  int iteration = 0;

  std::vector<cudaStream_t> streams;
  streams.reserve(max / (kThreads * kBlocks) + 2);

  for (; iteration < max; iteration += kThreads * kBlocks) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    ApplyKernel<<<kBlocks, kThreads, 0, stream>>>(std::forward<Args>(args)...,
                                                  offset_calculator(iteration));
    streams.emplace_back(stream);
  }
  if (excess >= kThreads) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    ApplyKernel<<<excess / kThreads, kThreads, 0, stream>>>(
        std::forward<Args>(args)..., offset_calculator(iteration));
    streams.emplace_back(stream);
  }
  iteration += (excess / kThreads) * kThreads;
  excess %= kThreads;
  if (excess > 0) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    ApplyKernel<<<1, excess, 0, stream>>>(std::forward<Args>(args)...,
                                          offset_calculator(iteration));
    streams.emplace_back(stream);
  }
  for (auto const& stream : streams) {
    SculptorCudaCheckError(cudaStreamSynchronize(stream));
    SculptorCudaCheckError(cudaStreamDestroy(stream));
  }
  streams.clear();
}
}  // namespace
void MatrixApplier::Apply(std::vector<glm::vec3>& vectors,
                          glm::mat4 const& matrix) {
  thrust::device_vector<glm::vec3> dvectors = vectors;
  auto dvectors_ptr =
      reinterpret_cast<float*>(thrust::raw_pointer_cast(dvectors.data()));

  SculptorCudaCheckError(
      cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0));

  Launch(
      vectors.size(), [](int iter) { return 3 * iter; }, dvectors_ptr);

  thrust::copy(dvectors.begin(), dvectors.end(), vectors.begin());
}

void MatrixApplier::Apply(cudaGraphicsResource* vectors,
                          int nvectors,
                          glm::mat4 const& matrix) {
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &vectors));
  float* dvectors = nullptr;
  size_t num_bytes;
  SculptorCudaCheckError(
      cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dvectors), &num_bytes, vectors));

  Launch(
      nvectors, [](int iter) { return 3 * iter; }, dvectors);

  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &vectors));
}
void MatrixApplier::Apply(cudaGraphicsResource* x,
                          cudaGraphicsResource* y,
                          cudaGraphicsResource* z,
                          int nvectors,
                          glm::mat4 const& matrix) {
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &x));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &y));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &z));

  float *dx, *dy, *dz;
  size_t num_bytes;
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dx), &num_bytes, x));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dy), &num_bytes, y));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dz), &num_bytes, z));
  SculptorCudaCheckError(
      cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0));

  Launch(
      nvectors, [](int off) { return off; }, dx, dy, dz);

  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &z));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &y));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &x));
}

std::unique_ptr<MatrixApplierBase> MatrixApplier::Clone() const {
  return std::make_unique<MatrixApplier>();
}
}  // namespace Sculptor
