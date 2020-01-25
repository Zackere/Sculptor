#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "matrix_applier_base.hpp"

namespace Sculptor {
class MatrixApplier : public MatrixApplierBase {
 public:
  ~MatrixApplier() override = default;

  void Apply(std::vector<glm::vec3>& vectors, glm::mat4 const& matrix) override;
  // void Apply(CudaGraphicsResource* vectors, glm::mat4 const& matrix)
  // override;
};
}  // namespace Sculptor
