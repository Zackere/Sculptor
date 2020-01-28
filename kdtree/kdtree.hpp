#pragma once

#include <glm/glm.hpp>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class KdTree {
 public:
  virtual ~KdTree() = default;

  virtual void Construct(CudaGraphicsResource<float>& x,
                         CudaGraphicsResource<float>& y,
                         CudaGraphicsResource<float>& z) = 0;

  virtual void RemoveNearest(
      CudaGraphicsResource<float>& kd_x,
      CudaGraphicsResource<float>& kd_y,
      CudaGraphicsResource<float>& kd_z,
      float threshold,
      CudaGraphicsResource<glm::vec3> const& query_points) = 0;
};
}  // namespace Sculptor
