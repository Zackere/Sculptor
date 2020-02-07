#include "kdtree_remover_base.hpp"

#include "../util/cudaCheckError.hpp"

namespace Sculptor {
std::vector<glm::vec3> KdTreeRemover::RemoveNearest(
    CudaGraphicsResource<float>& kd_x,
    CudaGraphicsResource<float>& kd_y,
    CudaGraphicsResource<float>& kd_z,
    CudaGraphicsResource<glm::vec3>& query_points,
    float threshold) {
  auto *x_res = kd_x.GetCudaResource(), *y_res = kd_y.GetCudaResource(),
       *z_res = kd_z.GetCudaResource();
  auto* query_res = query_points.GetCudaResource();

  SculptorCudaCheckError(cudaGraphicsMapResources(1, &x_res));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &y_res));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &z_res));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &query_res));

  float *dx, *dy, *dz, *dq;
  size_t num_bytes;
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dx), &num_bytes, x_res));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dy), &num_bytes, y_res));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dz), &num_bytes, z_res));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dq), &num_bytes, query_res));

  auto ret = remove_algorithm_->RemoveNearest(
      dx, dy, dz, kd_x.GetSize(), dq, query_points.GetSize(), threshold);
  kd_x.PopBack(ret.size());
  kd_y.PopBack(ret.size());
  kd_z.PopBack(ret.size());

  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &query_res));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &z_res));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &y_res));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &x_res));

  return ret;
}
}  // namespace Sculptor
