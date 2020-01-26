#pragma once

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <memory>
#include <string_view>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class ModelProviderBase;
class ShaderProviderBase;
class TextureProviderBase;
class MatrixApplierBase;

class glObject {
 public:
  glObject(std::unique_ptr<ModelProviderBase> model_provider,
           std::unique_ptr<ShaderProviderBase> shader_provider,
           std::unique_ptr<MatrixApplierBase> matrix_applier,
           std::unique_ptr<TextureProviderBase> texture_provider);
  ~glObject();

  void Render(glm::mat4 const& vp) const;
  void Transform(glm::mat4 const& m);
  void Enable() const;

  auto GetShader() const { return shader_; }
  auto GetTexture() const { return texture_; }
  auto GetNumberOfModelVertices() const {
    return model_parameters_.verticies->GetSize();
  }
  auto GetAvgPosition() const { return average_pos_; }

 private:
  struct {
    std::unique_ptr<CudaGraphicsResource<glm::vec3>> verticies, normals;
    std::unique_ptr<CudaGraphicsResource<glm::vec2>> uvs;
  } model_parameters_;
  GLuint shader_ = 0;
  GLuint vao_ = 0;
  GLuint texture_ = 0;
  std::unique_ptr<MatrixApplierBase> matrix_applier_;
  glm::vec3 average_pos_ = {};
};
}  // namespace Sculptor
