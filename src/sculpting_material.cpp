#include "../include/sculpting_material.hpp"

#include <algorithm>
#include <execution>
#include <functional>
#include <iostream>
#include <set>
#include <string_view>
#include <thread>
#include <utility>

#include "../external/lodepng/lodepng.cpp"
#include "../include/drill.hpp"
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

void InsertHollowCube(int size, std::vector<glm::vec3>& offsets, float step) {
  if (size <= 0)
    return;
  switch (size) {
    case 1:
      offsets.emplace_back(0, 0, 0);
      return;
    default:
      break;
  }
  auto start = step * (3 - size) / 2;
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
  side_len_ = 2.f / size;
  forward_ = {side_len_, 0, 0, 0};
  visible_instances_positions_.clear();
  invisible_instances_positions_.clear();
  switch (new_shape) {
    case InitialShape::CUBE:
      if (size <= 0)
        return;
      if (size == 1) {
        visible_instances_positions_.reserve(1);
        InsertHollowCube(size, visible_instances_positions_, side_len_);
        return;
      }
      visible_instances_positions_.reserve(size * size * size);
      InsertHollowCube(size, visible_instances_positions_, side_len_);
      invisible_instances_positions_.reserve((size - 2) * (size - 2) *
                                             (size - 2));
      for (int i = size - 2; i > 0; i -= 2)
        InsertHollowCube(i, invisible_instances_positions_, side_len_);
      break;
  }
  glBindBuffer(GL_ARRAY_BUFFER, visible_instances_positions_buffer_);
  glBufferData(GL_ARRAY_BUFFER,
               visible_instances_positions_.size() * 3 * sizeof(float),
               visible_instances_positions_.data(), GL_STATIC_DRAW);
}

void SculptingMaterial::Rotate(float amount) {
  auto m = glm::rotate(glm::mat4(1.f), amount, glm::vec3(0, 1, 0));
  Transform(m);
  forward_ = m * forward_;
}

void SculptingMaterial::Collide(Drill const& drill) {
  if (visible_instances_positions_.empty())
    return;
  std::thread t(
      [this]() { kd_tree_->Construct(invisible_instances_positions_); });
  kd_tree_->Construct(visible_instances_positions_);
  auto const& drill_vertices = drill.GetReferenceModelVertices();
  std::set<int, std::greater<int>> to_be_removed, to_be_added = {};
  for (auto const& x : drill_vertices) {
    auto p = kd_tree_->FindNearest(visible_instances_positions_, x);
    if (kd_tree_->GetDistanceToLastFound() < side_len_)
      to_be_removed.insert(p);
  }
  if (t.joinable())
    t.join();
  for (auto i : to_be_removed) {
    auto& p = visible_instances_positions_[i];

    to_be_added.insert(
        kd_tree_->Find(invisible_instances_positions_,
                       glm::vec3{p.x + forward_.x, p.y, p.z + forward_.z}));
    to_be_added.insert(
        kd_tree_->Find(invisible_instances_positions_,
                       glm::vec3{p.x - forward_.x, p.y, p.z + forward_.z}));
    to_be_added.insert(
        kd_tree_->Find(invisible_instances_positions_,
                       glm::vec3{p.x + forward_.x, p.y, p.z - forward_.z}));
    to_be_added.insert(
        kd_tree_->Find(invisible_instances_positions_,
                       glm::vec3{p.x - forward_.x, p.y, p.z - forward_.z}));
    to_be_added.insert(kd_tree_->Find(invisible_instances_positions_,
                                      glm::vec3{p.x, p.y + side_len_, p.z}));
    to_be_added.insert(kd_tree_->Find(invisible_instances_positions_,
                                      glm::vec3{p.x, p.y - side_len_, p.z}));

    p = visible_instances_positions_.back();
    visible_instances_positions_.pop_back();
  }
  to_be_added.erase(-1);
  for (auto const& i : to_be_added) {
    visible_instances_positions_.push_back(invisible_instances_positions_[i]);
    invisible_instances_positions_[i] = invisible_instances_positions_.back();
    invisible_instances_positions_.pop_back();
  }
  if (!to_be_removed.empty()) {
    glBindBuffer(GL_ARRAY_BUFFER, visible_instances_positions_buffer_);
    glBufferData(GL_ARRAY_BUFFER,
                 visible_instances_positions_.size() * 3 * sizeof(float),
                 visible_instances_positions_.data(), GL_STATIC_DRAW);
  }
}

void SculptingMaterial::Enable() const {
  glObject::Enable();
  auto materialOffsetsID = glGetAttribLocation(GetShader(), "offset");
  glEnableVertexAttribArray(materialOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, visible_instances_positions_buffer_);
  glVertexAttribPointer(materialOffsetsID, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void SculptingMaterial::Render(glm::mat4 const& vp) const {
  Enable();
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
  MatrixApplier::Apply(invisible_instances_positions_, m);
}
}  // namespace Sculptor
