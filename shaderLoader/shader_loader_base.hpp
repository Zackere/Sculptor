#pragma once

#include <string_view>

#include "GL/glew.h"

namespace Sculptor {
class ShaderLoaderBase {
 public:
  virtual ~ShaderLoaderBase() = default;

  virtual GLuint Load(std::string_view vertex_file_path,
                      std::string_view fragment_file_path) = 0;
};
}  // namespace Sculptor
