// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>

#include "light_base.hpp"

namespace Sculptor {
class ShaderProgramBase;

class DirectionalLight : public LightBase {
 public:
  DirectionalLight(glm::vec3 ambient,
                   glm::vec3 diffuse,
                   glm::vec3 specular,
                   glm::vec3 direction);
  ~DirectionalLight() override = default;

  void LoadIntoShader(ShaderProgramBase* shader) override;

  void SetPosition(glm::vec3) override {}

 private:
  static constexpr auto kClassName = "SculptorDirectionalLight";

  glm::vec3 direction_;
};
}  // namespace Sculptor
