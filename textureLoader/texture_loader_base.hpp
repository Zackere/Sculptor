#pragma once

#include <GL/glew.h>

#include <string_view>

namespace Sculptor {
class TextureLoaderBase {
 public:
  virtual ~TextureLoaderBase() = default;
  virtual GLuint Load(std::string_view path) = 0;
};
}  // namespace Sculptor
