// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <algorithm>
#include <glm/glm.hpp>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "kdtree_remover_base.hpp"

namespace Sculptor {
class KdTreeCPURemover : public KdTreeRemover::Algorithm {
 public:
  virtual ~KdTreeCPURemover() = default;

  std::vector<glm::vec3> RemoveNearest(float* x,
                                       float* y,
                                       float* z,
                                       int kd_size,
                                       float* query_points,
                                       int query_points_size,
                                       float threshold) override;

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
