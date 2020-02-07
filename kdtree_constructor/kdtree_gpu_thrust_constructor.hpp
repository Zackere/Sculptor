#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "kdtree_constructor_base.hpp"

namespace Sculptor {
class KdTreeGPUThrustConstructor : public KdTreeConstructor::Algorithm {
 public:
  virtual ~KdTreeGPUThrustConstructor() = default;

  void Construct(float* x, float* y, float* z, int size) override;

  // std::vector<glm::vec3> RemoveNearest(float* x,
  //                                      float* y,
  //                                      float* z,
  //                                      int kd_size,
  //                                      float* query_points,
  //                                      int query_points_size,
  //                                      float threshold) override;
};
}  // namespace Sculptor
