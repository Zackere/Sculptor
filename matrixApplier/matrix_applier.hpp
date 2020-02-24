// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "matrix_applier_base.hpp"

namespace Sculptor {
class MatrixApplier : public MatrixApplierBase {
 public:
  ~MatrixApplier() override = default;

  void Apply(std::vector<glm::vec3>& vectors, glm::mat4 const& matrix) override;
  void Apply(CudaGraphicsResource<glm::vec3>& vectors,
             glm::mat4 const& matrix) override;
  void Apply(CudaGraphicsResource<glm::mat4>& matricies,
             glm::mat4 const& matrix) override;
  void Apply(CudaGraphicsResource<float>& x,
             CudaGraphicsResource<float>& y,
             CudaGraphicsResource<float>& z,
             glm::mat4 const& matrix) override;
  std::unique_ptr<MatrixApplierBase> Clone() const override;
};
}  // namespace Sculptor
