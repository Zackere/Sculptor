#pragma once

#include <string>
#include <string_view>

#include "GL/glew.h"
#include "shader_provider_base.hpp"

namespace Sculptor {
class ShaderProvider : public ShaderProviderBase {
 public:
  ShaderProvider(std::string vertex_shader_path,
                 std::string fragment_shader_path);
  ~ShaderProvider() override = default;

  GLuint Get() override;

 private:
  std::string vertex_shader_path_;
  std::string fragment_shader_path_;
};
}  // namespace Sculptor
