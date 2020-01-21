#pragma once

#include <functional>
#include <map>
#include <vector>

#include "../include/kdtree.hpp"
#include "glm/glm.hpp"

namespace Sculptor {
class KdTreeGPU : public KdTree {
 public:
  ~KdTreeGPU() override = default;
  void Construct(std::vector<glm::vec3>& v) override;
  std::map<int, float, std::greater<>> FindNearest(
      std::vector<glm::vec3> const& kd_tree,
      std::vector<glm::vec3> const& query_points) override;
  int Find(std::vector<glm::vec3> const& kd_tree,
           glm::vec3 const& query_point) override;
};
}  // namespace Sculptor
