#include "../include/glObject.hpp"

#include <iostream>

#include "../include/objloader.hpp"
#include "../include/shader_loader.hpp"

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
  glBindBuffer(GL_ARRAY_BUFFER, GetVerticiesBuffer());
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, GetUVBuffer());
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, GetNormalsBuffer());
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void glObject::Render(glm::mat4 const& vp) const {
  glUseProgram(GetShader());
  auto mvp = vp * GetModelMatrix();
  glUniformMatrix4fv(glGetUniformLocation(GetShader(), "mvp"), 1, GL_FALSE,
                     &mvp[0][0]);
  glDrawArrays(GL_TRIANGLES, 0, reference_model_.verticies.size());
}
}  // namespace Sculptor
