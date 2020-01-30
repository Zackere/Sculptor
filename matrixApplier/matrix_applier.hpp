#pragma once

#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "matrix_applier_base.hpp"

namespace Sculptor {
class MatrixApplier : public MatrixApplierBase {
 public:
  ~MatrixApplier() override = default;

  void Apply(std::vector<glm::vec3>& vectors, glm::mat4 const& matrix) override;
  void Apply(cudaGraphicsResource* vectors,
             int nvectors,
             glm::mat4 const& matrix) override;
  void Apply(cudaGraphicsResource* x,
             cudaGraphicsResource* y,
             cudaGraphicsResource* z,
             int nvectors,
             glm::mat4 const& matrix) override;
  std::unique_ptr<MatrixApplierBase> Clone() const override;
};
}  // namespace Sculptor
