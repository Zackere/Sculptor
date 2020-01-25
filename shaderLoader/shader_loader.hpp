#pragma once

#include <string_view>

#include "GL/glew.h"
#include "shader_loader_base.hpp"

namespace Sculptor {
class ShaderLoader : public ShaderLoaderBase {
 public:
  ~ShaderLoader() override = default;

  GLuint Load(std::string_view vertex_file_path,
              std::string_view fragment_file_path) override;
};
}  // namespace Sculptor
