#include "../include/sculpting_material.hpp"

#include <iostream>
#include <string_view>

#include "../external/lodepng/lodepng.cpp"
#include "../include/matrix_applier.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Sculptor {
namespace {
constexpr const char* GetModelPath(SculptingMaterial::MaterialType type) {
  switch (type) {
    case SculptingMaterial::MaterialType::CUBE:
      return "../Sculptor/models/Cube.obj";
    default:
      return nullptr;
  }
}

constexpr const char* GetVertexShaderPath(
    SculptingMaterial::MaterialType type) {
  switch (type) {
    case SculptingMaterial::MaterialType::CUBE:
      return "../Sculptor/shaders/CubeShader.vs";
    default:
      return nullptr;
  }
}

constexpr const char* GetFragmentShaderPath(
    SculptingMaterial::MaterialType type) {
  switch (type) {
    case SculptingMaterial::MaterialType::CUBE:
      return "../Sculptor/shaders/CubeShader.fs";
    default:
      return nullptr;
  }
}

constexpr const char* GetTexturePath(SculptingMaterial::MaterialType type) {
  switch (type) {
    case SculptingMaterial::MaterialType::CUBE:
      return "../Sculptor/models/CubeTexture.png";
    default:
      return nullptr;
  }
}
}  // namespace
SculptingMaterial::SculptingMaterial(MaterialType material_type,
                                     InitialShape initial_shape,
                                     int size)
    : glObject(GetModelPath(material_type),
               GetVertexShaderPath(material_type),
               GetFragmentShaderPath(material_type)) {
  glGenBuffers(1, &offsets_buffer_);
  glVertexAttribDivisor(glGetAttribLocation(GetShader(), "offset"), 1);

  const auto scale = 1.f / size;
  Transform(glm::scale(glm::mat4(1.f), glm::vec3(scale, scale, scale)));

  std::vector<unsigned char> image;  // the raw pixels
  unsigned width, height;
  unsigned error =
      lodepng::decode(image, width, height, GetTexturePath(material_type));
  if (error)
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;
  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, image.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  Reset(initial_shape, size);
}

void SculptingMaterial::Reset(InitialShape new_shape, int size) {
  const auto step = 2.f / size;
  const auto start = step / 2 - 1, stop = -start;
  switch (new_shape) {
    case InitialShape::CUBE:
      offsets_.clear();
      offsets_.reserve(2 * size * size);
      for (auto x : {start, stop})
        for (auto y = start; y < 1.f; y += step)
          for (auto z = start; z < 1.f; z += step) {
            offsets_.emplace_back(x, y, z);
            offsets_.emplace_back(y, x, z);
            offsets_.emplace_back(y, z, x);
          }
      glBindBuffer(GL_ARRAY_BUFFER, offsets_buffer_);
      glBufferData(GL_ARRAY_BUFFER, offsets_.size() * 3 * sizeof(float),
                   offsets_.data(), GL_STATIC_DRAW);
      break;
  }
}

void SculptingMaterial::RemoveAt(unsigned index) {
  if (index >= offsets_.size()) {
    std::cerr << "Index is out of bounds : index: " << index << std::endl;
    return;
  }
  if (offsets_.empty())
    return;
  if (index + 1 < offsets_.size()) {
    offsets_[index] = offsets_.back();
    glBindBuffer(GL_ARRAY_BUFFER, offsets_buffer_);
    auto* p = reinterpret_cast<glm::vec3*>(
        glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
    p[index] = p[offsets_.size() - 1];
    glUnmapBuffer(GL_ARRAY_BUFFER);
  }
  offsets_.pop_back();
}

void SculptingMaterial::Rotate(float amount) {
  Transform(glm::rotate(glm::mat4(1.f), amount, glm::vec3(0, 1, 0)));
}

void SculptingMaterial::Enable() const {
  glObject::Enable();
  auto materialOffsetsID = glGetAttribLocation(GetShader(), "offset");
  glEnableVertexAttribArray(materialOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, GetMaterialElementsBuffer());
  glVertexAttribPointer(materialOffsetsID, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void SculptingMaterial::Render(glm::mat4 const& vp) const {
  glUseProgram(GetShader());
  glUniformMatrix4fv(glGetUniformLocation(GetShader(), "mvp"), 1, GL_FALSE,
                     &vp[0][0]);
  glBindTexture(GL_TEXTURE_2D, GetTexture());
  glDrawArraysInstanced(GL_TRIANGLES, 0, GetNVertices(),
                        GetMaterialElements().size());
}

void SculptingMaterial::Transform(glm::mat4 const& m) {
  glObject::Transform(m);
  MatrixApplier::Apply(&offsets_, m);
  glBindBuffer(GL_ARRAY_BUFFER, offsets_buffer_);
  glBufferData(GL_ARRAY_BUFFER, offsets_.size() * 3 * sizeof(float),
               offsets_.data(), GL_STATIC_DRAW);
}
}  // namespace Sculptor
