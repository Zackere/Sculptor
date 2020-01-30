#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class KdTree {
 public:
  virtual ~KdTree() = default;

  virtual void Construct(CudaGraphicsResource<float>& x,
                         CudaGraphicsResource<float>& y,
                         CudaGraphicsResource<float>& z) = 0;

  virtual std::vector<glm::vec3> RemoveNearest(
      CudaGraphicsResource<float>& kd_x,
      CudaGraphicsResource<float>& kd_y,
      CudaGraphicsResource<float>& kd_z,
      CudaGraphicsResource<glm::vec3>& query_points,
      float threshold,
      bool construct) = 0;
};
}  // namespace Sculptor
