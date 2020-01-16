#pragma once

#include <string_view>
#include <vector>

#include "GL/glew.h"
#include "glm/glm.hpp"

namespace Sculptor {
class glObject {
 public:
  glObject(std::string_view model_path,
           std::string_view vertex_shader_path,
           std::string_view fragment_shader_path);
  virtual ~glObject() = default;

  auto GetVerticiesBuffer() const { return reference_model_gl_.verticies; }
  auto GetUVBuffer() const { return reference_model_gl_.uvs; }
  auto GetNormalsBuffer() const { return reference_model_gl_.normals; }
  auto GetNVertices() const { return reference_model_.verticies.size(); }
  auto GetShader() const { return shader_; }

  virtual void Enable() const;
  virtual void Render(glm::mat4 const& vp) const;

 protected:
  struct {
    std::vector<glm::vec3> verticies = {};
    std::vector<glm::vec2> uvs = {};
    std::vector<glm::vec3> normals = {};
  } reference_model_ = {};
  struct {
    GLuint verticies = 0, uvs = 0, normals = 0;
  } reference_model_gl_ = {};
  GLuint shader_ = 0;
};
}  // namespace Sculptor
