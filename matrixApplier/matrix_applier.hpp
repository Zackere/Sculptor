#pragma once

#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <vector>

#include "matrix_applier_base.hpp"

namespace Sculptor {
class MatrixApplier : public MatrixApplierBase {
 public:
  ~MatrixApplier() override = default;

  void Apply(cudaGraphicsResource* vectors,
             int nvectors,
             glm::mat4 const& matrix) override;
  void Apply(cudaGraphicsResource* x,
             cudaGraphicsResource* y,
             cudaGraphicsResource* z,
             int nvectors,
             glm::mat4 const& matrix) override;
};
}  // namespace Sculptor
