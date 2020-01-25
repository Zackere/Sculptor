#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "GL/glew.h"
#include "glm/glm.hpp"

namespace Sculptor {
class ObjLoaderBase;
class ShaderLoaderBase;
class MatrixApplierBase;

class glObject {
 public:
  glObject(std::string_view model_path,
           std::string_view vertex_shader_path,
           std::string_view fragment_shader_path,
           std::unique_ptr<ObjLoaderBase> obj_loader,
           std::unique_ptr<ShaderLoaderBase> shader_loader,
           std::unique_ptr<MatrixApplierBase> matrix_applier);

  void Render(glm::mat4 const& vp) const;

  auto const& GetModelVertices() const { return model_parameters_.verticies; }

  void Enable() const;
  void Transform(glm::mat4 const& m);
  auto GetNumberOfModelVerticies() const {
    return model_parameters_.verticies.size();
  }

 private:
  struct {
    std::vector<glm::vec3> verticies = {};
    std::vector<glm::vec2> uvs = {};
    std::vector<glm::vec3> normals = {};
  } model_parameters_ = {};
  struct {
    GLuint verticies = 0, uvs = 0, normals = 0;
  } model_parameters_gl_ = {};
  GLuint shader_ = 0;
  GLuint vao_ = 0;
  std::unique_ptr<MatrixApplierBase> matrix_applier_ = nullptr;
};
}  // namespace Sculptor
