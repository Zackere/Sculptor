#include "../include/sculpting_material.hpp"

#include <algorithm>
#include <execution>
#include <iostream>
#include <string_view>
#include <utility>

#include "../external/lodepng/lodepng.cpp"
#include "../include/kdtree_cpu.hpp"
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

void InsertHollowCube(int size, std::vector<glm::vec3>& offsets) {
  if (size <= 0)
    return;
  switch (size) {
    case 1:
      offsets.emplace_back(0, 0, 0);
      return;
    case 2:
      for (auto x : {-0.5f, 0.5f})
        for (auto y : {-0.5f, 0.5f})
          for (auto z : {-0.5f, 0.5f})
            offsets.emplace_back(x, y, z);
      return;
    default:
      break;
  }
  const auto step = 2.f / size;
  auto start = -1.f + 3 * step / 2;
  float y, z;
  for (auto x : {start - step, -start + step}) {
    y = start;
    for (int j = 2; j < size; ++j, y += step) {
      z = start;
      for (int k = 2; k < size; ++k, z += step) {
        offsets.emplace_back(x, y, z);
        offsets.emplace_back(y, x, z);
        offsets.emplace_back(y, z, x);
      }
      offsets.emplace_back(y, x, x);
      offsets.emplace_back(y, x, -x);
      offsets.emplace_back(x, y, x);
      offsets.emplace_back(x, y, -x);
      offsets.emplace_back(x, x, y);
      offsets.emplace_back(x, -x, y);
    }
    offsets.emplace_back(x, x, x);
    offsets.emplace_back(-x, x, x);
    offsets.emplace_back(x, -x, x);
    offsets.emplace_back(x, x, -x);
  }
}
}  // namespace
SculptingMaterial::SculptingMaterial(MaterialType material_type,
                                     InitialShape initial_shape,
                                     int size,
                                     std::unique_ptr<KdTree> kd_tree)
    : glObject(GetModelPath(material_type),
               GetVertexShaderPath(material_type),
               GetFragmentShaderPath(material_type)),
      kd_tree_(std::move(kd_tree)) {
  glGenBuffers(1, &visible_instances_positions_buffer_);
  glVertexAttribDivisor(glGetAttribLocation(GetShader(), "offset"), 1);

  const auto scale = 1.f / size;
  Transform(glm::scale(glm::mat4(1.f), glm::vec3(scale, scale, scale)));

  std::vector<unsigned char> image_raw;
  unsigned width, height;
  unsigned error =
      lodepng::decode(image_raw, width, height, GetTexturePath(material_type));
  if (error)
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;
  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, image_raw.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  Reset(initial_shape, size);
}

void SculptingMaterial::Reset(InitialShape new_shape, int size) {
  visible_instances_positions_.clear();
  invisible_instances_positions_.clear();
  switch (new_shape) {
    case InitialShape::CUBE:
      if (size <= 0)
        return;
      if (size == 1) {
        InsertHollowCube(size, visible_instances_positions_);
        return;
      }
      visible_instances_positions_.reserve(6 * size * (size - 2) +
                                           8);  // size ^ 3 - (size - 2) ^ 3
      InsertHollowCube(size, visible_instances_positions_);
      invisible_instances_positions_.reserve((size - 2) * (size - 2) *
                                             (size - 2));
      for (int i = size - 2; i > 0; i -= 2)
        InsertHollowCube(i, invisible_instances_positions_);
      break;
  }
  glBindBuffer(GL_ARRAY_BUFFER, visible_instances_positions_buffer_);
  glBufferData(GL_ARRAY_BUFFER,
               visible_instances_positions_.size() * 3 * sizeof(float),
               visible_instances_positions_.data(), GL_STATIC_DRAW);
}

void SculptingMaterial::Rotate(float amount) {
  Transform(glm::rotate(glm::mat4(1.f), amount, glm::vec3(0, 1, 0)));
}

void SculptingMaterial::Enable() const {
  glObject::Enable();
  auto materialOffsetsID = glGetAttribLocation(GetShader(), "offset");
  glEnableVertexAttribArray(materialOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, visible_instances_positions_buffer_);
  glVertexAttribPointer(materialOffsetsID, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void SculptingMaterial::Render(glm::mat4 const& vp) const {
  glUseProgram(GetShader());
  glUniformMatrix4fv(glGetUniformLocation(GetShader(), "mvp"), 1, GL_FALSE,
                     &vp[0][0]);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glDrawArraysInstanced(GL_TRIANGLES, 0, GetNumberOfReferenceModelVerticies(),
                        visible_instances_positions_.size());
}

void SculptingMaterial::Transform(glm::mat4 const& m) {
  glObject::Transform(m);
  MatrixApplier::Apply(visible_instances_positions_, m);
  glBindBuffer(GL_ARRAY_BUFFER, visible_instances_positions_buffer_);
  glBufferData(GL_ARRAY_BUFFER,
               visible_instances_positions_.size() * 3 * sizeof(float),
               visible_instances_positions_.data(), GL_STATIC_DRAW);
}
}  // namespace Sculptor
