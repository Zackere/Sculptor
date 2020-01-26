#pragma once

#include <GL/glew.h>

#include <string>
#include <string_view>

#include "shader_provider_base.hpp"

namespace Sculptor {
class ShaderProvider : public ShaderProviderBase {
 public:
  ShaderProvider(std::string_view vertex_shader_path,
                 std::string_view fragment_shader_path);
  ~ShaderProvider() override = default;

  GLuint Get() override;

 private:
  std::string vertex_shader_path_;
  std::string fragment_shader_path_;
};
}  // namespace Sculptor
