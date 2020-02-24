// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>

#include "light_base.hpp"

namespace Sculptor {
class ShaderProgramBase;

class Spotlight : public LightBase {
 public:
  Spotlight(glm::vec3 ambient,
            glm::vec3 diffuse,
            glm::vec3 specular,
            glm::vec3 position,
            glm::vec3 look_target,
            glm::vec2 cutoff_);
  ~Spotlight() override = default;

  void LoadIntoShader(ShaderProgramBase* shader) override;
  void LookAt(glm::vec3 target);
  void SetPosition(glm::vec3 pos) override;

 private:
  static constexpr auto kClassName = "SculptorSpotlight";

  glm::vec3 position_;
  glm::vec3 look_target_;
  glm::vec2 cutoff_;
};
}  // namespace Sculptor
