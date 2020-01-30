#include "kdtree.hpp"

namespace Sculptor {

void KdTreeConstructor::Construct(CudaGraphicsResource<float>& x,
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

  construction_algorithm_->Construct(dx, dy, dz, x.GetSize());

  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);
}

std::vector<glm::vec3> KdTreeRemover::RemoveNearest(
    CudaGraphicsResource<float>& kd_x,
    CudaGraphicsResource<float>& kd_y,
    CudaGraphicsResource<float>& kd_z,
    CudaGraphicsResource<glm::vec3>& query_points,
    float threshold) {
  auto *x_res = kd_x.GetCudaResource(), *y_res = kd_y.GetCudaResource(),
       *z_res = kd_z.GetCudaResource();
  auto* query_res = query_points.GetCudaResource();

  cudaGraphicsMapResources(1, &x_res);
  cudaGraphicsMapResources(1, &y_res);
  cudaGraphicsMapResources(1, &z_res);
  cudaGraphicsMapResources(1, &query_res);

  float *dx, *dy, *dz, *dq;
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dx),
                                       &num_bytes, x_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dy),
                                       &num_bytes, y_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dz),
                                       &num_bytes, z_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dq),
                                       &num_bytes, query_res);

  auto ret = remove_algorithm_->RemoveNearest(
      dx, dy, dz, kd_x.GetSize(), dq, query_points.GetSize(), threshold);
  kd_x.PopBack(ret.size());
  kd_y.PopBack(ret.size());
  kd_z.PopBack(ret.size());

  cudaGraphicsUnmapResources(1, &query_res);
  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);

  return ret;
}

}  // namespace Sculptor
