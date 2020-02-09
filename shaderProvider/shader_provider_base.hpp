// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

namespace Sculptor {
class ShaderProviderBase {
 public:
  virtual ~ShaderProviderBase() = default;

  virtual GLuint Get() = 0;
};
}  // namespace Sculptor
