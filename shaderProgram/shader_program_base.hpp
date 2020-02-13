// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

namespace Sculptor {
class ShaderProgramBase {
 public:
  virtual ~ShaderProgramBase() = default;

  virtual GLuint Get() = 0;
};
}  // namespace Sculptor
