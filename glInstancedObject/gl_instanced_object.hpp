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

  int GetNumberOfInstances() const { return model_transforms_.GetSize(); }
  CudaGraphicsResource<glm::mat4>& GetModelTransforms() {
    return model_transforms_;
  }

  void AddInstances(std::vector<glm::vec3> const& instances);

  void SetShader(std::unique_ptr<ShaderProgramBase> shader);
  ShaderProgramBase* GetShader();

 private:
  std::unique_ptr<glObject> reference_model_;
  std::unique_ptr<ShapeGeneratorBase> shape_generator_;
  CudaGraphicsResource<glm::mat4> model_transforms_;
  std::unique_ptr<MatrixApplierBase> matrix_applier_;
};
}  // namespace Sculptor
