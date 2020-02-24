// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>

#include "light_base.hpp"

namespace Sculptor {
class ShaderProgramBase;

class PointLight : public LightBase {
 public:
  PointLight(glm::vec3 ambient,
             glm::vec3 diffuse,
             glm::vec3 specular,
             glm::vec3 position,
             glm::vec3 attenuation);
  ~PointLight() override = default;

  void LoadIntoShader(ShaderProgramBase* shader) override;
  void SetPosition(glm::vec3 pos) override;

 private:
  static constexpr auto kClassName = "SculptorPointLight";

  glm::vec3 position_;
  glm::vec3 attenuation_;
};
}  // namespace Sculptor
