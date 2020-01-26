#include "png_texture_provider.hpp"

#include <iostream>
#include <vector>

#include "../external/lodepng/lodepng.hpp"

namespace Sculptor {
GLuint PNGTextureProvider::Get() {
  GLuint texture = 0;
  std::vector<unsigned char> image_raw;
  unsigned width, height;
  unsigned error = lodepng::decode(image_raw, width, height, path_.data());
  if (error)
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, image_raw.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  return texture;
}
}  // namespace Sculptor
