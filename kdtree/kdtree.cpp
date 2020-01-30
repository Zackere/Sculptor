#include "kdtree.hpp"

#include "../util/cudaCheckError.hpp"

namespace Sculptor {

std::future<void> KdTreeConstructor::Construct(cudaGraphicsResource* kd_x,
                                               cudaGraphicsResource* kd_y,
                                               cudaGraphicsResource* kd_z,
                                               int size) {
  float *dx, *dy, *dz;
  size_t num_bytes;
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dx), &num_bytes, kd_x));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dy), &num_bytes, kd_y));
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dz), &num_bytes, kd_z));

  return std::async(std::launch::async, [=]() {
    construction_algorithm_->Construct(dx, dy, dz, size);
  });
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

std::tuple<cudaGraphicsResource*, cudaGraphicsResource*, cudaGraphicsResource*>
KdTreeConstructor::LoadResources(CudaGraphicsResource<float>& x,
                                 CudaGraphicsResource<float>& y,
                                 CudaGraphicsResource<float>& z) {
  auto kd_x = x.GetCudaResource();
  auto kd_y = y.GetCudaResource();
  auto kd_z = z.GetCudaResource();

  SculptorCudaCheckError(cudaGraphicsMapResources(1, &kd_x));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &kd_y));
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &kd_z));

  return {kd_x, kd_y, kd_z};
}

void KdTreeConstructor::UnloadResources(cudaGraphicsResource* kd_x,
                                        cudaGraphicsResource* kd_y,
                                        cudaGraphicsResource* kd_z) {
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &kd_z));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &kd_y));
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &kd_x));
}

}  // namespace Sculptor
