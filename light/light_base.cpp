// Copyright 2020 Wojciech Replin. All rights reserved.

#include "light_base.hpp"

#include "../shaderProgram/shader_program_base.hpp"

namespace Sculptor {
std::map<std::string, std::set<unsigned>> LightBase::taken_ids_{};

LightBase::LightBase(glm::vec3 ambient,
                     glm::vec3 diffuse,
                     glm::vec3 specular,
                     std::string light_class_name)
    : light_class_name_(light_class_name),
      ambient_(ambient),
      diffuse_(diffuse),
      specular_(specular) {
  for (auto id : taken_ids_[light_class_name_])
    if (id == id_)
      ++id_;
    else
      break;
  taken_ids_[light_class_name_].insert(id_);
}

void LightBase::LoadIntoShader(ShaderProgramBase* shader) {
  glUseProgram(shader->Get());
  auto id_string = std::to_string(id_);
  glUniform3f(glGetUniformLocation(
                  shader->Get(),
                  (light_class_name_ + "[" + id_string + "].ambient").c_str()),
              ambient_.x, ambient_.y, ambient_.z);
  glUniform3f(glGetUniformLocation(
                  shader->Get(),
                  (light_class_name_ + "[" + id_string + "].diffuse").c_str()),
              diffuse_.x, diffuse_.y, diffuse_.z);
  glUniform3f(glGetUniformLocation(
                  shader->Get(),
                  (light_class_name_ + "[" + id_string + "].specular").c_str()),
              specular_.x, specular_.y, specular_.z);
  Enable(shader);
}

void LightBase::UnloadFromShader(ShaderProgramBase* shader) {
  Disable(shader);
}

void LightBase::Enable(ShaderProgramBase* shader) {
  glUseProgram(shader->Get());
  glUniform1i(glGetUniformLocation(
                  shader->Get(),
                  (light_class_name_ + '[' + std::to_string(id_) + "].enabled")
                      .c_str()),
              true);
}

void LightBase::Disable(ShaderProgramBase* shader) {
  glUseProgram(shader->Get());
  glUniform1i(glGetUniformLocation(
                  shader->Get(),
                  (light_class_name_ + '[' + std::to_string(id_) + "].enabled")
                      .c_str()),
              false);
}

LightBase::~LightBase() = default;

}  // namespace Sculptor
