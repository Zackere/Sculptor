// Copyright 2020 Wojciech Replin. All rights reserved.

#include "gl_instanced_object.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>
#include <vector>

#include "../glObject/gl_object.hpp"
#include "../matrixApplier/matrix_applier_base.hpp"
#include "../modelProvider/model_provider_base.hpp"
#include "../shaderProgram/shader_program_base.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"
#include "../textureProvider/texture_provider_base.hpp"

namespace Sculptor {
glInstancedObject::glInstancedObject(
    int ninstances_init,
    int ninstances_max,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<ShapeGeneratorBase> shape_generator,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : reference_model_(std::move(reference_model)),
      shape_generator_(std::move(shape_generator)),
      model_transforms_(ninstances_max),
      matrix_applier_(std::move(matrix_applier)) {
  std::vector<glm::mat4> model_transforms;
  model_transforms.reserve(
      shape_generator_->GetNumberOfOutputs(ninstances_init));
  for (auto& v : shape_generator_->Generate(ninstances_init))
    model_transforms.emplace_back(glm::translate(glm::mat4(1.f), v));

  model_transforms_.SetData(model_transforms.data(), model_transforms.size());

  auto materialTransformsID = glGetAttribLocation(
      reference_model_->GetShader()->Get(), "model_transform");
  glEnableVertexAttribArray(materialTransformsID);
  glBindBuffer(GL_ARRAY_BUFFER, model_transforms_.GetGLBuffer());
  for (int i = 0; i < 4; ++i) {
    glEnableVertexAttribArray(materialTransformsID + i);
    glVertexAttribPointer(materialTransformsID + i, 4, GL_FLOAT, GL_FALSE,
                          sizeof(glm::mat4),
                          reinterpret_cast<void*>(sizeof(float) * i * 4));
    glVertexAttribDivisor(materialTransformsID + i, 1);
  }
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
}

void glInstancedObject::AddInstances(std::vector<glm::vec3> const& instances) {
  for (auto& v : instances)
    model_transforms_.PushBack(glm::translate(glm::mat4(1.f), v));
}

void glInstancedObject::SetShader(std::unique_ptr<ShaderProgramBase> shader) {
  reference_model_->SetShader(std::move(shader));
}

ShaderProgramBase* glInstancedObject::GetShader() {
  return reference_model_->GetShader();
}

}  // namespace Sculptor
