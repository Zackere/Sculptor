#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <utility>

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

  void Construct(CudaGraphicsResource<float>& x,
                 CudaGraphicsResource<float>& y,
                 CudaGraphicsResource<float>& z);

 private:
  std::unique_ptr<Algorithm> construction_algorithm_ = nullptr;
};
}  // namespace Sculptor
