#pragma once

#include <memory>
#include <vector>

#include "glm/glm.hpp"

namespace Sculptor {
class MatrixApplierBase {
 public:
  virtual ~MatrixApplierBase() = default;

  virtual void Apply(std::vector<glm::vec3>& vectors,
                     glm::mat4 const& matrix) = 0;
};
}  // namespace Sculptor
