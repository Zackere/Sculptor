// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

namespace Sculptor {
class ShaderProgramBase {
 public:
  virtual ~ShaderProgramBase() = default;

  virtual void Use() = 0;
  virtual GLuint GetUniformLocation(char const* name) = 0;
  virtual GLuint GetAttribLocation(char const* name) = 0;
};
}  // namespace Sculptor
