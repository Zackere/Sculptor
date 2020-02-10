// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>

#include <string>
#include <string_view>

#include "texture_provider_base.hpp"

namespace Sculptor {
class PNGTextureProvider : public TextureProviderBase {
 public:
  explicit PNGTextureProvider(std::string_view path) : path_(path) {}
  ~PNGTextureProvider() override = default;
  GLuint Get() override;

 private:
  std::string path_ = {};
};
}  // namespace Sculptor
