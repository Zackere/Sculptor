#pragma once

#include <vector>

#include "glm/glm.hpp"

namespace Sculptor {
class KdTree {
 public:
  virtual ~KdTree() = default;
  virtual void Construct(std::vector<glm::vec3>& v) = 0;
  virtual int FindNearest(std::vector<glm::vec3> const& kd_tree,
                          glm::vec3 const& query_point) = 0;
  virtual int Find(std::vector<glm::vec3> const& kd_tree,
                   glm::vec3 const& query_point) = 0;
  virtual float GetDistanceToLastFound() const = 0;
};
}  // namespace Sculptor
