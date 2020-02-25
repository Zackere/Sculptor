// Copyright 2020 Wojciech Replin. All rights reserved.

#include "gl_instanced_object.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>
#include <vector>

#include "../glObject/gl_object.hpp"
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
      ti_model_transforms_(ninstances_max),
      matrix_applier_(std::move(matrix_applier)) {
  auto assign_transforms = [this](char const* name,
                                  CudaGraphicsResource<glm::mat4>& buf) {
    auto id = glGetAttribLocation(reference_model_->GetShader()->Get(), name);
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
  assign_transforms("ti_model_transform", ti_model_transforms_);
}

glInstancedObject::~glInstancedObject() = default;

void glInstancedObject::Render(glm::mat4 const& vp) const {
  reference_model_->Enable();
  glUniformMatrix4fv(
      glGetUniformLocation(reference_model_->GetShader()->Get(), "vp"), 1,
      GL_FALSE, &vp[0][0]);
  glBindTexture(GL_TEXTURE_2D, reference_model_->GetTexture());
  glDrawArraysInstanced(GL_TRIANGLES, 0,
                        reference_model_->GetNumberOfModelVertices(),
                        GetNumberOfInstances());
}

void glInstancedObject::Transform(glm::mat4 const& m) {
  matrix_applier_->Apply(model_transforms_, m);
  matrix_applier_->Apply(ti_model_transforms_, glm::transpose(glm::inverse(m)));
}

void glInstancedObject::AddInstances(std::vector<glm::vec3> const& instances) {
  for (auto& v : instances) {
    model_transforms_.PushBack(glm::translate(glm::mat4(1.f), v));
    ti_model_transforms_.PushBack(glm::mat4(1.f));
  }
}

void glInstancedObject::SetShader(std::unique_ptr<ShaderProgramBase> shader) {
  reference_model_->SetShader(std::move(shader));
}

ShaderProgramBase* glInstancedObject::GetShader() {
  return reference_model_->GetShader();
}

}  // namespace Sculptor
