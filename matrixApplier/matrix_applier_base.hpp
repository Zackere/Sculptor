#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class MatrixApplierBase {
 public:
  virtual ~MatrixApplierBase() = default;

  virtual void Apply(std::vector<glm::vec3>& vectors,
                     glm::mat4 const& matrix) = 0;
  // virtual void Apply(CudaGraphicsResource* vectors,
  //                    glm::mat4 const& matrix) = 0;
};
}  // namespace Sculptor
