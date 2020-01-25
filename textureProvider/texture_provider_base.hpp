#pragma once

#include <GL/glew.h>

#include <string_view>

namespace Sculptor {
class TextureProviderBase {
 public:
  virtual ~TextureProviderBase() = default;
  virtual GLuint Get() = 0;
};
}  // namespace Sculptor
