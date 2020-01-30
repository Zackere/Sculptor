#pragma once

#include <future>
#include <glm/glm.hpp>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class KdTreeConstructor {
 public:
  class Algorithm {
   public:
    virtual ~Algorithm() = default;
    virtual void Construct(float* x, float* y, float* z, int size) = 0;
  };
  KdTreeConstructor(std::unique_ptr<Algorithm> alg)
      : construction_algorithm_(std::move(alg)) {}

  [[nodiscard]] std::future<void> Construct(cudaGraphicsResource* kd_x,
                                            cudaGraphicsResource* kd_y,
                                            cudaGraphicsResource* kd_z,
                                            int size);

  [[nodiscard]] std::
      tuple<cudaGraphicsResource*, cudaGraphicsResource*, cudaGraphicsResource*>
      LoadResources(CudaGraphicsResource<float>& kd_x,
                    CudaGraphicsResource<float>& kd_y,
                    CudaGraphicsResource<float>& kd_z);
  void UnloadResources(cudaGraphicsResource* kd_x,
                       cudaGraphicsResource* kd_y,
                       cudaGraphicsResource* kd_z);

 private:
  std::unique_ptr<Algorithm> construction_algorithm_ = nullptr;
};

class KdTreeRemover {
 public:
  class Algorithm {
   public:
    virtual ~Algorithm() = default;
    virtual std::vector<glm::vec3> RemoveNearest(float* x,
                                                 float* y,
                                                 float* z,
                                                 int kd_size,
                                                 float* query_points,
                                                 int query_points_size,
                                                 float threshold) = 0;
  };
  KdTreeRemover(std::unique_ptr<Algorithm> alg)
      : remove_algorithm_(std::move(alg)) {}

  std::vector<glm::vec3> RemoveNearest(
      CudaGraphicsResource<float>& kd_x,
      CudaGraphicsResource<float>& kd_y,
      CudaGraphicsResource<float>& kd_z,
      CudaGraphicsResource<glm::vec3>& query_points,
      float threshold);

 private:
  std::unique_ptr<Algorithm> remove_algorithm_ = nullptr;
};
}  // namespace Sculptor
