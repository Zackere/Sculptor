// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

namespace Sculptor {
class ShaderProgramBase {
 public:
  enum class LightModel {
    BLINN,
    PHONG,
  };

  virtual ~ShaderProgramBase() = default;

  virtual void Use() = 0;
  virtual GLuint GetUniformLocation(char const* name) = 0;
  virtual GLuint GetAttribLocation(char const* name) = 0;
  virtual void SetLightModel(LightModel light_model) = 0;
};
}  // namespace Sculptor
