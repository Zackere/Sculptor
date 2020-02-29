// Copyright 2020 Wojciech Replin. All rights reserved.

#include "point_light.hpp"

#include <string>

#include "../shaderProgram/shader_program_base.hpp"

namespace Sculptor {
PointLight::PointLight(glm::vec3 ambient,
                       glm::vec3 diffuse,
                       glm::vec3 specular,
                       glm::vec3 position,
                       glm::vec3 attenuation)
    : LightBase(ambient, diffuse, specular, kClassName),
      position_(position),
      attenuation_(attenuation) {}

void PointLight::LoadIntoShader(ShaderProgramBase* shader) {
  LightBase::LoadIntoShader(shader);
  shader->Use();
  auto id = std::to_string(GetId());
  glUniform3f(shader->GetUniformLocation(
                  (std::string(kClassName) + '[' + id + "].position").c_str()),
              position_.x, position_.y, position_.z);
  glUniform3f(
      shader->GetUniformLocation(
          (std::string(kClassName) + '[' + id + "].attenuation").c_str()),
      attenuation_.x, attenuation_.y, attenuation_.z);
}

void PointLight::SetPosition(glm::vec3 pos) {
  position_ = pos;
}
}  // namespace Sculptor
