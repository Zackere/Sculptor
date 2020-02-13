// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>

namespace Sculptor {
class ShaderProgramBase;

class Camera {
 public:
  virtual ~Camera() = default;
  virtual glm::mat4 GetTransform() = 0;
  virtual glm::vec3 SetPos(glm::vec3 const& pos) = 0;
  virtual glm::vec3 Zoom(float amount) = 0;
  virtual glm::vec3 Rotate(glm::vec2 const& direction) = 0;
  virtual void LoadIntoShader(ShaderProgramBase* shader) = 0;
};
}  // namespace Sculptor
