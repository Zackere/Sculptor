#pragma once

#include <glm/glm.hpp>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "kdtree.hpp"

namespace Sculptor {
class KdTreeGPU : public KdTree {
 public:
  virtual ~KdTreeGPU() = default;

  void Construct(CudaGraphicsResource<float>& x,
                 CudaGraphicsResource<float>& y,
                 CudaGraphicsResource<float>& z) override;

  void RemoveNearest(
      CudaGraphicsResource<float>& kd_x,
      CudaGraphicsResource<float>& kd_y,
      CudaGraphicsResource<float>& kd_z,
      float threshold,
      CudaGraphicsResource<glm::vec3> const& query_points) override;
};
}  // namespace Sculptor