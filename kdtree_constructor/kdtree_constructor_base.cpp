#include "kdtree_constructor_base.hpp"

#include "../util/cudaCheckError.hpp"

namespace Sculptor {
void KdTreeConstructor::Construct(CudaGraphicsResource<float>& x,
                                  CudaGraphicsResource<float>& y,
                                  CudaGraphicsResource<float>& z) {
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

  construction_algorithm_->Construct(dx, dy, dz, x.GetSize());

  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &z_res));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &y_res));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &x_res));
}
}  // namespace Sculptor
