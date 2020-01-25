#include "gl_object.hpp"

#include <utility>

#include "../matrixApplier/matrix_applier_base.hpp"
#include "../modelProvider/model_provider_base.hpp"
#include "../shaderProvider/shader_provider_base.hpp"
#include "../textureProvider/texture_provider_base.hpp"

namespace Sculptor {
glObject::glObject(std::unique_ptr<ModelProviderBase> model_provider,
                   std::unique_ptr<ShaderProviderBase> shader_provider,
                   std::unique_ptr<MatrixApplierBase> matrix_applier,
                   std::unique_ptr<TextureProviderBase> texture_provider)
    : matrix_applier_(std::move(matrix_applier)) {
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);
  glGenBuffers(1, &model_parameters_gl_.verticies);
  glGenBuffers(1, &model_parameters_gl_.uvs);
  glGenBuffers(1, &model_parameters_gl_.normals);

  model_provider->Get(model_parameters_.verticies, model_parameters_.uvs,
                      model_parameters_.normals);
  shader_ = shader_provider->Get();
  texture_ = texture_provider->Get();

  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_gl_.verticies);
  glBufferData(GL_ARRAY_BUFFER,
               model_parameters_.verticies.size() * 3 * sizeof(float),
               model_parameters_.verticies.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_gl_.uvs);
  glBufferData(GL_ARRAY_BUFFER,
               model_parameters_.uvs.size() * 2 * sizeof(float),
               model_parameters_.uvs.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_gl_.normals);
  glBufferData(GL_ARRAY_BUFFER,
               model_parameters_.normals.size() * 3 * sizeof(float),
               model_parameters_.normals.data(), GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_gl_.verticies);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_gl_.uvs);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_gl_.normals);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void glObject::Enable() const {
  glUseProgram(shader_);
  glBindVertexArray(vao_);
}

void glObject::Render(glm::mat4 const& vp) const {
  Enable();
  glUniformMatrix4fv(glGetUniformLocation(shader_, "mvp"), 1, GL_FALSE,
                     &vp[0][0]);
  glDrawArrays(GL_TRIANGLES, 0, model_parameters_.verticies.size());
}

void glObject::Transform(glm::mat4 const& m) {
  matrix_applier_->Apply(model_parameters_.verticies, m);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_gl_.verticies);
  glBufferData(GL_ARRAY_BUFFER,
               model_parameters_.verticies.size() * 3 * sizeof(float),
               model_parameters_.verticies.data(), GL_STATIC_DRAW);
  matrix_applier_->Apply(model_parameters_.normals,
                         glm::transpose(glm::inverse(m)));
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_gl_.normals);
  glBufferData(GL_ARRAY_BUFFER,
               model_parameters_.normals.size() * 3 * sizeof(float),
               model_parameters_.normals.data(), GL_STATIC_DRAW);
}
}  // namespace Sculptor
