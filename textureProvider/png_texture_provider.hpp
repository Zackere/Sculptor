#pragma once

#include <GL/glew.h>

#include <string>
#include <string_view>

#include "texture_provider_base.hpp"

namespace Sculptor {
class PNGTextureProvider : public TextureProviderBase {
 public:
  PNGTextureProvider(std::string path) : path_(path) {}
  ~PNGTextureProvider() override = default;
  GLuint Get() override;

 private:
  std::string path_;
};
}  // namespace Sculptor
