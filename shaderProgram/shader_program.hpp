// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

#include <string>
#include <string_view>

#include "shader_program_base.hpp"

namespace Sculptor {
class ShaderProgram : public ShaderProgramBase {
 public:
  ShaderProgram(std::string_view vertex_shader_path,
                std::string_view fragment_shader_path);
  ~ShaderProgram() override;

  GLuint Get() override;

 private:
  GLuint program_ = 0;

  ShaderProgram(ShaderProgram const&) = delete;
  ShaderProgram& operator=(ShaderProgram const&) = delete;

  ShaderProgram(ShaderProgram&&) = default;
  ShaderProgram& operator=(ShaderProgram&&) = default;
};
}  // namespace Sculptor
