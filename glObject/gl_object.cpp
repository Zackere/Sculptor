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

  std::vector<glm::vec3> verticies, normals;
  std::vector<glm::vec2> uvs;
  model_provider->Get(verticies, uvs, normals);

  model_parameters_.verticies =
      std::make_unique<CudaGraphicsResource<glm::vec3>>(verticies.size());
  model_parameters_.verticies->SetData(verticies.data(), verticies.size());

  model_parameters_.uvs =
      std::make_unique<CudaGraphicsResource<glm::vec2>>(uvs.size());
  model_parameters_.uvs->SetData(uvs.data(), uvs.size());

  model_parameters_.normals =
      std::make_unique<CudaGraphicsResource<glm::vec3>>(normals.size());
  model_parameters_.normals->SetData(normals.data(), normals.size());

  shader_ = shader_provider->Get();
  if (texture_provider)
    texture_ = texture_provider->Get();

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_.verticies->GetGLBuffer());
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_.uvs->GetGLBuffer());
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, model_parameters_.normals->GetGLBuffer());
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
  glDrawArrays(GL_TRIANGLES, 0, GetNumberOfModelVertices());
}

void glObject::Transform(glm::mat4 const& m) {
  matrix_applier_->Apply(model_parameters_.verticies->GetCudaResource(),
                         model_parameters_.verticies->GetSize(), m);
  matrix_applier_->Apply(model_parameters_.normals->GetCudaResource(),
                         model_parameters_.normals->GetSize(),
                         glm::transpose(glm::inverse(m)));
}
}  // namespace Sculptor
