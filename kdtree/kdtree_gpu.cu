// clang-format off
#include "kdtree_gpu.hpp"
// clang-format on

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

namespace Sculptor {
namespace {
void ConstructRecursive(thrust::device_ptr<float> x,
                        thrust::device_ptr<float> y,
                        thrust::device_ptr<float> z,
                        int begin,
                        int end) {
  if (end <= begin)
    return;
  auto mid = begin + (end - begin) / 2;
  auto zip = thrust::make_zip_iterator(thrust::make_tuple(y, z));
  thrust::sort_by_key(thrust::device, x + begin, x + end, zip + begin);
  ConstructRecursive(y, z, x, begin, mid);
  ConstructRecursive(y, z, x, mid + 1, end);
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

  float *dx, *dy, *dz;
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dx),
                                       &num_bytes, x_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dy),
                                       &num_bytes, y_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dz),
                                       &num_bytes, z_res);

  ConstructRecursive(thrust::device_ptr<float>(dx),
                     thrust::device_ptr<float>(dy),
                     thrust::device_ptr<float>(dz), 0, x.GetSize());

  cudaGraphicsUnmapResources(1, &x_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &z_res);
}

void KdTreeGPU::RemoveNearest(CudaGraphicsResource<float>&,
                              CudaGraphicsResource<float>&,
                              CudaGraphicsResource<float>&,
                              float,
                              CudaGraphicsResource<glm::vec3> const&) {}
}  // namespace Sculptor
