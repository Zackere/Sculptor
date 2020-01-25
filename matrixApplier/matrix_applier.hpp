#pragma once

#include <vector>

#include "glm/glm.hpp"
#include "matrix_applier_base.hpp"

namespace Sculptor {
class MatrixApplier : public MatrixApplierBase {
 public:
  ~MatrixApplier() override = default;

  void Apply(std::vector<glm::vec3>& vectors, glm::mat4 const& matrix) override;
};
}  // namespace Sculptor
