#include "glObject.hpp"

#include <utility>

#include "../matrixApplier/matrix_applier_base.hpp"
#include "../objLoader/obj_loader_base.hpp"
#include "../shaderLoader/shader_loader_base.hpp"

namespace Sculptor {
glObject::glObject(std::string_view model_path,
                   std::string_view vertex_shader_path,
                   std::string_view fragment_shader_path,
                   std::unique_ptr<ObjLoaderBase> obj_loader,
                   std::unique_ptr<ShaderLoaderBase> shader_loader,
                   std::unique_ptr<MatrixApplierBase> matrix_applier)
    : matrix_applier_(std::move(matrix_applier)) {
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  obj_loader->Load(model_path.data(), model_parameters_.verticies,
                   model_parameters_.uvs, model_parameters_.normals);

  glGenBuffers(1, &model_parameters_gl_.verticies);
  glGenBuffers(1, &model_parameters_gl_.uvs);
  glGenBuffers(1, &model_parameters_gl_.normals);

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

  shader_ = shader_loader->Load(vertex_shader_path.data(),
                                fragment_shader_path.data());
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
