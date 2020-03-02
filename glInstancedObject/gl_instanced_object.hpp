// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "../glObject/gl_object.hpp"

namespace Sculptor {
class MatrixApplierBase;
class ShaderProgramBase;

class glInstancedObject {
 public:
  glInstancedObject(int ninstances_max,
                    std::unique_ptr<glObject> reference_model,
                    std::unique_ptr<MatrixApplierBase> matrix_applier);
  ~glInstancedObject();

  void Render(glm::mat4 const& vp) const;
  void Transform(glm::mat4 const& m);
  template <typename T>
  void Load(T* t) {
    reference_model_->Load(t);
  }
  template <typename T>
  void Unload(T* t) {
    reference_model_->Unload(t);
  }

  int GetNumberOfInstances() const { return model_transforms_.GetSize(); }
  CudaGraphicsResource<glm::mat4>& GetModelTransforms() {
    return model_transforms_;
  }

  int AddInstance(glm::mat4 const& instance);
  void PopInstance();
  unsigned SetInstance(glm::mat4 const& new_instance, unsigned index);
  glm::mat4 GetTransformAt(unsigned index);
  glm::mat4 GetGlobalTransform() const { return global_transform_; }
  glObject& GetObject() { return *reference_model_; }

  std::unique_ptr<ShaderProgramBase> SetShader(
      std::unique_ptr<ShaderProgramBase> shader);

 private:
  std::unique_ptr<glObject> reference_model_;
  CudaGraphicsResource<glm::mat4> model_transforms_;
  CudaGraphicsResource<glm::mat4> i_model_transforms_;
  std::unique_ptr<MatrixApplierBase> matrix_applier_;

  glm::mat4 global_transform_ = glm::mat4(1.f);
};
}  // namespace Sculptor
