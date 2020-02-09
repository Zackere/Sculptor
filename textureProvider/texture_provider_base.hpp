// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

namespace Sculptor {
class TextureProviderBase {
 public:
  virtual ~TextureProviderBase() = default;
  virtual GLuint Get() = 0;
};
}  // namespace Sculptor
