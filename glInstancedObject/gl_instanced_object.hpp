// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class MatrixApplierBase;
class ShapeGeneratorBase;
class glObject;
class ShaderProgramBase;

class glInstancedObject {
 public:
  glInstancedObject(int ninstances_init,
                    int ninstances_max,
                    std::unique_ptr<glObject> reference_model,
                    std::unique_ptr<ShapeGeneratorBase> shape_generator,
                    std::unique_ptr<MatrixApplierBase> matrix_applier);
  ~glInstancedObject();

  void Render(glm::mat4 const& vp) const;
  void Transform(glm::mat4 const& m);

  int GetNumberOfInstances() const { return x_positions_buffer_.GetSize(); }
  CudaGraphicsResource<float>& GetVecticesX() { return x_positions_buffer_; }
  CudaGraphicsResource<float>& GetVecticesY() { return y_positions_buffer_; }
  CudaGraphicsResource<float>& GetVecticesZ() { return z_positions_buffer_; }

  void AddInstances(std::vector<glm::vec3> const& instances);

  void SetShader(std::unique_ptr<ShaderProgramBase> shader);

 private:
  std::unique_ptr<glObject> reference_model_;
  std::unique_ptr<ShapeGeneratorBase> shape_generator_;
  CudaGraphicsResource<float> x_positions_buffer_;
  CudaGraphicsResource<float> y_positions_buffer_;
  CudaGraphicsResource<float> z_positions_buffer_;
  std::unique_ptr<MatrixApplierBase> matrix_applier_;
};
}  // namespace Sculptor
