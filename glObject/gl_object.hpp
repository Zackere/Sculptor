#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "GL/glew.h"
#include "glm/glm.hpp"

namespace Sculptor {
class ModelProviderBase;
class ShaderProviderBase;
class MatrixApplierBase;
class TextureProviderBase;

class glObject {
 public:
  glObject(std::unique_ptr<ModelProviderBase> model_provider,
           std::unique_ptr<ShaderProviderBase> shader_provider,
           std::unique_ptr<MatrixApplierBase> matrix_applier,
           std::unique_ptr<TextureProviderBase> texture_provider);

  void Render(glm::mat4 const& vp) const;
  void Transform(glm::mat4 const& m);
  void Enable() const;

  auto GetShader() const { return shader_; }
  auto GetTexture() const { return texture_; }
  auto GetNumberOfModelVertices() const {
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
  GLuint texture_ = 0;
  std::unique_ptr<MatrixApplierBase> matrix_applier_ = nullptr;
};
}  // namespace Sculptor
