// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <memory>
#include <string_view>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class ModelProviderBase;
class TextureProviderBase;
class MatrixApplierBase;
class ShaderProgramBase;

class glObject {
 public:
  glObject(std::unique_ptr<ModelProviderBase> model_provider,
           std::unique_ptr<ShaderProgramBase> shader_program,
           std::unique_ptr<MatrixApplierBase> matrix_applier,
           std::unique_ptr<TextureProviderBase> texture_provider,
           glm::vec4 light_coefficient);
  ~glObject();

  void Render(glm::mat4 const& vp) const;
  void Transform(glm::mat4 const& m);
  void Enable() const;

  auto GetNumberOfModelVertices() const {
    return model_parameters_.verticies->GetSize();
  }
  auto GetAvgPosition() const { return average_pos_; }
  auto* GetVertices() { return model_parameters_.verticies.get(); }

  std::unique_ptr<ShaderProgramBase> SetShader(
      std::unique_ptr<ShaderProgramBase> shader);
  ShaderProgramBase* GetShader();

  template <typename T>
  void Load(T* t) {
    t->LoadIntoShader(shader_.get());
  }
  template <typename T>
  void Unload(T* t) {
    t->UnloadFromShader(shader_.get());
  }

 private:
  struct {
    std::unique_ptr<CudaGraphicsResource<glm::vec3>> verticies, normals;
    std::unique_ptr<CudaGraphicsResource<glm::vec2>> uvs;
  } model_parameters_;
  std::unique_ptr<ShaderProgramBase> shader_;
  GLuint vao_ = 0;
  GLuint texture_ = 0;
  std::unique_ptr<MatrixApplierBase> matrix_applier_;
  glm::vec3 average_pos_ = {};
  glm::vec4 light_coefficient_ = {};
};
}  // namespace Sculptor
