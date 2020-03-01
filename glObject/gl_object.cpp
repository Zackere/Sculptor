// Copyright 2020 Wojciech Replin. All rights reserved.

#include "gl_object.hpp"

#include <execution>
#include <numeric>
#include <utility>

#include "../matrixApplier/matrix_applier_base.hpp"
#include "../modelProvider/model_provider_base.hpp"
#include "../shaderProgram/shader_program_base.hpp"
#include "../textureProvider/texture_provider_base.hpp"

namespace Sculptor {
glObject::glObject(std::unique_ptr<ModelProviderBase> model_provider,
                   std::unique_ptr<ShaderProgramBase> shader_program,
                   std::unique_ptr<MatrixApplierBase> matrix_applier,
                   std::unique_ptr<TextureProviderBase> texture_provider,
                   glm::vec4 light_coefficient)
    : model_parameters_{nullptr, nullptr, nullptr},
      shader_(nullptr),
      matrix_applier_(std::move(matrix_applier)),
      light_coefficient_(light_coefficient) {
  glGenVertexArrays(1, &vao_);

  std::vector<glm::vec3> verticies, normals;
  std::vector<glm::vec2> uvs;
  model_provider->Get(verticies, uvs, normals);
  average_pos_ =
      std::reduce(std::execution::par, verticies.begin(), verticies.end()) /
      static_cast<float>(verticies.size());

  model_parameters_.verticies =
      std::make_unique<CudaGraphicsResource<glm::vec3>>(verticies.size());
  model_parameters_.verticies->SetData(verticies.data(), verticies.size());

  model_parameters_.uvs =
      std::make_unique<CudaGraphicsResource<glm::vec2>>(uvs.size());
  model_parameters_.uvs->SetData(uvs.data(), uvs.size());

  model_parameters_.normals =
      std::make_unique<CudaGraphicsResource<glm::vec3>>(normals.size());
  model_parameters_.normals->SetData(normals.data(), normals.size());

  texture_ = texture_provider->Get();

  SetShader(std::move(shader_program));
}

glObject::~glObject() {
  glDeleteTextures(1, &texture_);
  glDeleteVertexArrays(1, &vao_);
}

void glObject::Enable() const {
  shader_->Use();
  glBindVertexArray(vao_);
  glBindTexture(GL_TEXTURE_2D, texture_);
}

std::unique_ptr<ShaderProgramBase> glObject::SetShader(
    std::unique_ptr<ShaderProgramBase> shader) {
  auto ret = std::move(shader_);
  shader_ = std::move(shader);

  shader_->Use();
  glBindVertexArray(vao_);

  auto id = shader_->GetAttribLocation("vertex_position");
  glEnableVertexAttribArray(id);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_.verticies->GetGLBuffer());
  glVertexAttribPointer(id, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  id = shader_->GetAttribLocation("vertex_uv");
  glEnableVertexAttribArray(id);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_.uvs->GetGLBuffer());
  glVertexAttribPointer(id, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

  id = shader_->GetAttribLocation("vertex_normal");
  glEnableVertexAttribArray(id);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_.normals->GetGLBuffer());
  glVertexAttribPointer(id, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  glUniform4f(shader_->GetUniformLocation("light_coefficient"),
              light_coefficient_.x, light_coefficient_.y, light_coefficient_.z,
              light_coefficient_.w);

  return ret;
}

ShaderProgramBase* glObject::GetShader() {
  return shader_.get();
}

void glObject::Render(glm::mat4 const& vp) const {
  Enable();
  glUniformMatrix4fv(shader_->GetUniformLocation("vp"), 1, GL_FALSE, &vp[0][0]);
  glDrawArrays(GL_TRIANGLES, 0, GetNumberOfModelVertices());
}

void glObject::Transform(glm::mat4 const& m) {
  matrix_applier_->Apply(*model_parameters_.verticies, m);
  matrix_applier_->Apply(*model_parameters_.normals,
                         glm::transpose(glm::inverse(m)));
  average_pos_ = m * glm::vec4(average_pos_, 1.f);
}
}  // namespace Sculptor
