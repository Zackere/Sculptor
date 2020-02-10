// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>

namespace Sculptor {
class Camera {
 public:
  virtual ~Camera() = default;
  virtual glm::mat4 GetTransform() = 0;
  virtual void SetPos(glm::vec3 const& pos) = 0;
  virtual void Move(glm::vec3 const& direction) = 0;
  virtual void Rotate(glm::vec2 const& direction) = 0;
};
}  // namespace Sculptor
