#pragma once

#include <GL/glew.h>

#include <string_view>

namespace Sculptor {
class ShaderProviderBase {
 public:
  virtual ~ShaderProviderBase() = default;

  virtual GLuint Get() = 0;
};
}  // namespace Sculptor
