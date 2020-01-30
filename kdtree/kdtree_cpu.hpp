#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "kdtree.hpp"

namespace Sculptor {
class KdTreeCPU : public KdTree {
 public:
  virtual ~KdTreeCPU() = default;

  void Construct(CudaGraphicsResource<float>& x,
                 CudaGraphicsResource<float>& y,
                 CudaGraphicsResource<float>& z) override;

  std::vector<glm::vec3> RemoveNearest(
      CudaGraphicsResource<float>& kd_x,
      CudaGraphicsResource<float>& kd_y,
      CudaGraphicsResource<float>& kd_z,
      CudaGraphicsResource<glm::vec3>& query_points,
      float threshold,
      bool construct) override;

 private:
  glm::vec3 query_point_ = {0, 0, 0};
  std::vector<glm::vec3>::iterator closest_node_ =
      std::vector<glm::vec3>::iterator();
  float best_distance_ = 0.f;

  float DistFromQuery(glm::vec3 const& v) {
    return std::max(std::abs(v.x - query_point_.x),
                    std::max(std::abs(v.y - query_point_.y),
                             std::abs(v.z - query_point_.z)));
  }
  void FindNearestRecursive(std::vector<glm::vec3>::iterator begin,
                            std::vector<glm::vec3>::iterator end,
                            int level);
};
}  // namespace Sculptor
