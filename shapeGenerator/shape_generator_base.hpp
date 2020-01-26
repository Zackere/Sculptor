#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace Sculptor {
class ShapeGeneratorBase {
 public:
  virtual ~ShapeGeneratorBase() = default;
  virtual std::vector<glm::vec3> Generate(int) = 0;
  virtual int GetNumberOfOutputs(int) = 0;
};
}  // namespace Sculptor
