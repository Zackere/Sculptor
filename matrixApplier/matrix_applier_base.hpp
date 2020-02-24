// Copyright 2020 Wojciech Replin. All rights reserved.

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
  virtual void Apply(CudaGraphicsResource<glm::vec3>& vectors,
                     glm::mat4 const& matrix) = 0;
  virtual void Apply(CudaGraphicsResource<glm::mat4>& matricies,
                     glm::mat4 const& matrix) = 0;
  virtual void Apply(CudaGraphicsResource<float>& x,
                     CudaGraphicsResource<float>& y,
                     CudaGraphicsResource<float>& z,
                     glm::mat4 const& matrix) = 0;
  virtual std::unique_ptr<MatrixApplierBase> Clone() const = 0;
};
}  // namespace Sculptor
