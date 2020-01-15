#pragma once

#include <string_view>

#include "GL/glew.h"

namespace ShaderLoader {
GLuint Load(std::string_view vertex_file_path,
            std::string_view fragment_file_path);
}  // namespace ShaderLoader
