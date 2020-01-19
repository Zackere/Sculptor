#include "../include/glObject.hpp"

#include "../include/matrix_applier.hpp"
#include "../include/objloader.hpp"
#include "../include/shader_loader.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Sculptor {
glObject::glObject(std::string_view model_path,
                   std::string_view vertex_shader_path,
                   std::string_view fragment_shader_path) {
  glGenBuffers(1, &reference_model_gl_.verticies);
  glGenBuffers(1, &reference_model_gl_.uvs);
  glGenBuffers(1, &reference_model_gl_.normals);

  OBJLoader::LoadOBJ(model_path.data(), reference_model_.verticies,
                     reference_model_.uvs, reference_model_.normals);

  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.verticies);
  glBufferData(GL_ARRAY_BUFFER,
               reference_model_.verticies.size() * 3 * sizeof(float),
               reference_model_.verticies.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.uvs);
  glBufferData(GL_ARRAY_BUFFER, reference_model_.uvs.size() * 2 * sizeof(float),
               reference_model_.uvs.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.normals);
  glBufferData(GL_ARRAY_BUFFER,
               reference_model_.normals.size() * 3 * sizeof(float),
               reference_model_.normals.data(), GL_STATIC_DRAW);

  shader_ = ShaderLoader::Load(vertex_shader_path.data(),
                               fragment_shader_path.data());
}

void glObject::Enable() const {
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.verticies);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.uvs);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.normals);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void glObject::Render(glm::mat4 const& vp) const {
  Enable();
  glUseProgram(shader_);
  glUniformMatrix4fv(glGetUniformLocation(shader_, "mvp"), 1, GL_FALSE,
                     &vp[0][0]);
  glDrawArrays(GL_TRIANGLES, 0, reference_model_.verticies.size());
}

void glObject::Transform(glm::mat4 const& m) {
  MatrixApplier::Apply(reference_model_.verticies, m);
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.verticies);
  glBufferData(GL_ARRAY_BUFFER,
               reference_model_.verticies.size() * 3 * sizeof(float),
               reference_model_.verticies.data(), GL_STATIC_DRAW);
  MatrixApplier::Apply(reference_model_.normals,
                       glm::transpose(glm::inverse(m)));
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.normals);
  glBufferData(GL_ARRAY_BUFFER,
               reference_model_.normals.size() * 3 * sizeof(float),
               reference_model_.normals.data(), GL_STATIC_DRAW);
}
}  // namespace Sculptor
