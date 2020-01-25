#pragma once

#include <vector>

#include "glm/glm.hpp"

namespace Sculptor {
class ShapeGeneratorBase {
 public:
  virtual ~ShapeGeneratorBase() = default;
  virtual std::vector<glm::vec3> Generate(int) = 0;
};
}  // namespace Sculptor
