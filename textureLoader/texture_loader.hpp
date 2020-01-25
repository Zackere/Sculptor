#pragma once

#include <GL/glew.h>

#include <string_view>

#include "texture_loader_base.hpp"

namespace Sculptor {
class TextureLoader : public TextureLoaderBase {
 public:
  ~TextureLoader() override = default;
  GLuint Load(std::string_view path) override;
};
}  // namespace Sculptor
