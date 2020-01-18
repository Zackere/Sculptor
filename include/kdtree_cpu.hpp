#pragma once

#include "../include/kdtree.hpp"

namespace Sculptor {
class KdTreeCPU : public KdTree {
 public:
  ~KdTreeCPU() override = default;
  void Construct(std::vector<glm::vec3>& v) override;
};
}  // namespace Sculptor
