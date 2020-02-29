// Copyright 2020 Wojciech Replin. All rights reserved.

#include "spotlight.hpp"

#include <string>

#include "../shaderProgram/shader_program_base.hpp"

namespace Sculptor {
Spotlight::Spotlight(glm::vec3 ambient,
                     glm::vec3 diffuse,
                     glm::vec3 specular,
                     glm::vec3 position,
                     glm::vec3 look_target,
                     glm::vec2 cutoff)
    : LightBase(ambient, diffuse, specular, kClassName),
      position_(position),
      look_target_(look_target),
      cutoff_(cutoff) {}

void Spotlight::LoadIntoShader(ShaderProgramBase* shader) {
  LightBase::LoadIntoShader(shader);
  shader->Use();
  auto id = std::to_string(GetId());
  glUniform3f(shader->GetUniformLocation(
                  (std::string(kClassName) + '[' + id + "].position").c_str()),
              position_.x, position_.y, position_.z);
  glUniform3f(
      shader->GetUniformLocation(
          (std::string(kClassName) + '[' + id + "].look_target").c_str()),
      look_target_.x, look_target_.y, look_target_.z);
  glUniform2f(shader->GetUniformLocation(
                  (std::string(kClassName) + '[' + id + "].cutoff").c_str()),
              cutoff_.x, cutoff_.y);
}

void Spotlight::LookAt(glm::vec3 target) {
  look_target_ = target;
}

void Spotlight::SetPosition(glm::vec3 pos) {
  position_ = pos;
}
}  // namespace Sculptor
