// Copyright 2020 Wojciech Replin. All rights reserved.

#include "gl_instanced_object.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>
#include <vector>

#include "../matrixApplier/matrix_applier_base.hpp"
#include "../modelProvider/model_provider_base.hpp"
#include "../shaderProgram/shader_program_base.hpp"
#include "../textureProvider/texture_provider_base.hpp"

namespace Sculptor {
glInstancedObject::glInstancedObject(
    int ninstances_max,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : reference_model_(std::move(reference_model)),
      model_transforms_(ninstances_max),
      i_model_transforms_(ninstances_max),
      matrix_applier_(std::move(matrix_applier)) {
  auto assign_transforms = [this](char const* name,
                                  CudaGraphicsResource<glm::mat4>& buf) {
    auto id = reference_model_->GetShader()->GetAttribLocation(name);
    glEnableVertexAttribArray(id);
    glBindBuffer(GL_ARRAY_BUFFER, buf.GetGLBuffer());
    for (int i = 0; i < 4; ++i) {
      glEnableVertexAttribArray(id + i);
      glVertexAttribPointer(id + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                            reinterpret_cast<void*>(sizeof(float) * i * 4));
      glVertexAttribDivisor(id + i, 1);
    }
  };

  assign_transforms("model_transform", model_transforms_);
  assign_transforms("i_model_transform", i_model_transforms_);
}

glInstancedObject::~glInstancedObject() = default;

void glInstancedObject::Render(glm::mat4 const& vp) const {
  reference_model_->Enable();
  glUniformMatrix4fv(reference_model_->GetShader()->GetUniformLocation("vp"), 1,
                     GL_FALSE, &vp[0][0]);
  glUniformMatrix4fv(
      reference_model_->GetShader()->GetUniformLocation("global_transform"), 1,
      GL_FALSE, &global_transform_[0][0]);
  auto i_global_transform = glm::inverse(global_transform_);
  glUniformMatrix4fv(
      reference_model_->GetShader()->GetUniformLocation("i_global_transform"),
      1, GL_FALSE, &i_global_transform[0][0]);
  glDrawArraysInstanced(GL_TRIANGLES, 0,
                        reference_model_->GetNumberOfModelVertices(),
                        GetNumberOfInstances());
}

void glInstancedObject::Transform(glm::mat4 const& m) {
  global_transform_ = m * global_transform_;
}

int glInstancedObject::AddInstance(const glm::mat4& instance) {
  model_transforms_.PushBack(instance);
  i_model_transforms_.PushBack(glm::inverse(instance));
  return model_transforms_.GetSize() - 1;
}

void glInstancedObject::PopInstance() {
  model_transforms_.PopBack();
  i_model_transforms_.PopBack();
}

unsigned glInstancedObject::SetInstance(glm::mat4 const& new_instance,
                                        unsigned index) {
  model_transforms_.Set(new_instance, index);
  i_model_transforms_.Set(glm::inverse(new_instance), index);
  return index;
}

glm::mat4 glInstancedObject::GetTransformAt(unsigned index) {
  return model_transforms_.Get(index);
}

std::unique_ptr<ShaderProgramBase> glInstancedObject::SetShader(
    std::unique_ptr<ShaderProgramBase> shader) {
  return reference_model_->SetShader(std::move(shader));
}
}  // namespace Sculptor
