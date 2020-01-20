#pragma once

#include <functional>
#include <map>
#include <vector>

#include "glm/glm.hpp"

namespace Sculptor {
class KdTree {
 public:
  virtual ~KdTree() = default;
  virtual void Construct(std::vector<glm::vec3>& v) = 0;
  virtual std::map<int, float, std::greater<int>> FindNearest(
      std::vector<glm::vec3> const& kd_tree,
      std::vector<glm::vec3> const& query_points) = 0;
  virtual int Find(std::vector<glm::vec3> const& kd_tree,
                   glm::vec3 const& query_point) = 0;
  virtual float GetDistanceToLastFound() const = 0;
};
}  // namespace Sculptor
