// Copyright 2020 Wojciech Replin. All rights reserved.

#include "directional_light.hpp"

#include <string>

#include "../shaderProgram/shader_program_base.hpp"

namespace Sculptor {
DirectionalLight::DirectionalLight(glm::vec3 ambient,
                                   glm::vec3 diffuse,
                                   glm::vec3 specular,
                                   glm::vec3 direction)
    : LightBase(ambient, diffuse, specular, kClassName),
      direction_(direction) {}

void DirectionalLight::LoadIntoShader(ShaderProgramBase* shader) {
  LightBase::LoadIntoShader(shader);
  shader->Use();
  auto id = std::to_string(GetId());
  glUniform3f(shader->GetUniformLocation(
                  (std::string(kClassName) + '[' + id + "].direction").c_str()),
              direction_.x, direction_.y, direction_.z);
}

}  // namespace Sculptor
