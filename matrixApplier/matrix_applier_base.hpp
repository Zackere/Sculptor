#pragma once

#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <vector>

namespace Sculptor {
class MatrixApplierBase {
 public:
  virtual ~MatrixApplierBase() = default;

  virtual void Apply(cudaGraphicsResource* vectors,
                     int nvectors,
                     glm::mat4 const& matrix) = 0;
  virtual void Apply(cudaGraphicsResource* x,
                     cudaGraphicsResource* y,
                     cudaGraphicsResource* z,
                     int nvectors,
                     glm::mat4 const& matrix) = 0;
};
}  // namespace Sculptor
