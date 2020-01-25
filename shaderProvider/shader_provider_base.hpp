#pragma once

#include <string_view>

#include "GL/glew.h"

namespace Sculptor {
class ShaderProviderBase {
 public:
  virtual ~ShaderProviderBase() = default;

  virtual GLuint Get() = 0;
};
}  // namespace Sculptor
