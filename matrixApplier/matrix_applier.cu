// Copyright 2020 Wojciech Replin. All rights reserved.

#include "matrix_applier.hpp"
// clang-format on

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <utility>

#include "../util/cudaCheckError.hpp"

namespace Sculptor {
namespace {
constexpr int kMatrixSize = 4;
constexpr int kThreads = 256;
constexpr int kBlocks = 64;

__constant__ float c_matrix[kMatrixSize][kMatrixSize];

__global__ void ApplyKernel(float3* vectors3, int ninstance) {
  __shared__ float s_data[3 * kThreads];
  float* vectors = reinterpret_cast<float*>(vectors3);
  int index = 3 * (blockDim.x * blockIdx.x + ninstance) + threadIdx.x;
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

__global__ void ApplyKernel(float4* vectors4, int ninstance) {
  __shared__ float s_data[4 * kThreads];
  float* vectors = reinterpret_cast<float*>(vectors4);
  int index = 4 * (blockDim.x * blockIdx.x + ninstance) + threadIdx.x;
  s_data[threadIdx.x] = vectors[index];
  s_data[threadIdx.x + blockDim.x] = vectors[index + blockDim.x];
  s_data[threadIdx.x + 2 * blockDim.x] = vectors[index + 2 * blockDim.x];
  s_data[threadIdx.x + 3 * blockDim.x] = vectors[index + 3 * blockDim.x];
  __syncthreads();
  float4 v = reinterpret_cast<float4*>(s_data)[threadIdx.x];
  reinterpret_cast<float4*>(s_data)[threadIdx.x] =
      float4{v.x * c_matrix[0][0] + v.y * c_matrix[1][0] +
                 v.z * c_matrix[2][0] + v.w * c_matrix[3][0],
             v.x * c_matrix[0][1] + v.y * c_matrix[1][1] +
                 v.z * c_matrix[2][1] + v.w * c_matrix[3][1],
             v.x * c_matrix[0][2] + v.y * c_matrix[1][2] +
                 v.z * c_matrix[2][2] + v.w * c_matrix[3][2],
             v.x * c_matrix[0][3] + v.y * c_matrix[1][3] +
                 v.z * c_matrix[2][3] + v.w * c_matrix[3][3]};
  __syncthreads();
  vectors[index] = s_data[threadIdx.x];
  vectors[index + blockDim.x] = s_data[threadIdx.x + blockDim.x];
  vectors[index + 2 * blockDim.x] = s_data[threadIdx.x + 2 * blockDim.x];
  vectors[index + 3 * blockDim.x] = s_data[threadIdx.x + 3 * blockDim.x];
}

__global__ void ApplyKernel(float* x, float* y, float* z, int ninstance) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 3 * ninstance;
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

template <typename... LaunchArgs>
void Launch(int datasize, LaunchArgs&&... args) {
  int excess = datasize % (kThreads * kBlocks);
  int max = datasize - excess;
  int iteration = 0;

  std::vector<cudaStream_t> streams;
  streams.reserve(max / (kThreads * kBlocks) + 2);

  for (; iteration < max; iteration += kThreads * kBlocks) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    ApplyKernel<<<kBlocks, kThreads, 0, stream>>>(
        std::forward<LaunchArgs>(args)..., iteration);
    streams.emplace_back(stream);
  }
  if (excess >= kThreads) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    ApplyKernel<<<excess / kThreads, kThreads, 0, stream>>>(
        std::forward<LaunchArgs>(args)..., iteration);
    streams.emplace_back(stream);
  }
  iteration += (excess / kThreads) * kThreads;
  excess %= kThreads;
  if (excess > 0) {
    cudaStream_t stream;
    SculptorCudaCheckError(cudaStreamCreate(&stream));
    ApplyKernel<<<1, excess, 0, stream>>>(std::forward<LaunchArgs>(args)...,
                                          iteration);
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
  auto* dvectors_ptr =
      reinterpret_cast<float3*>(thrust::raw_pointer_cast(dvectors.data()));

  SculptorCudaCheckError(
      cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0));

  Launch(vectors.size(), dvectors_ptr);

  thrust::copy(dvectors.begin(), dvectors.end(), vectors.begin());
}

void MatrixApplier::Apply(CudaGraphicsResource<glm::vec3>& vectors,
                          glm::mat4 const& matrix) {
  auto* resource = vectors.GetCudaResource();
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &resource));
  float3* dvectors = nullptr;
  size_t num_bytes;
  SculptorCudaCheckError(
      cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dvectors), &num_bytes, resource));

  Launch(vectors.GetSize(), dvectors);

  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &resource));
}

void MatrixApplier::Apply(CudaGraphicsResource<glm::mat4>& matricies,
                          glm::mat4 const& matrix) {
  auto* resource = matricies.GetCudaResource();
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &resource));
  float4* dvectors = nullptr;
  size_t num_bytes;
  SculptorCudaCheckError(
      cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dvectors), &num_bytes, resource));

  Launch(4 * matricies.GetSize(), dvectors);

  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &resource));
}

void MatrixApplier::Apply(CudaGraphicsResource<float>& x,
                          CudaGraphicsResource<float>& y,
                          CudaGraphicsResource<float>& z,
                          glm::mat4 const& matrix) {
  auto *x_res = x.GetCudaResource(), *y_res = y.GetCudaResource(),
       *z_res = z.GetCudaResource();
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &x_res));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &y_res));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &z_res));

  float *dx, *dy, *dz;
  size_t num_bytes;
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dx), &num_bytes, x_res));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dy), &num_bytes, y_res));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dz), &num_bytes, z_res));
  SculptorCudaCheckError(
      cudaMemcpyToSymbol(c_matrix, &matrix, sizeof(c_matrix), 0));

  Launch(x.GetSize(), dx, dy, dz);

  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &z_res));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &y_res));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &x_res));
}

std::unique_ptr<MatrixApplierBase> MatrixApplier::Clone() const {
  return std::make_unique<MatrixApplier>();
}
}  // namespace Sculptor
