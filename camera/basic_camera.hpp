// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>

#include "camera_base.hpp"

namespace Sculptor {
class BasicCamera : public Camera {
 public:
  BasicCamera(glm::vec3 const& initial_pos,
              glm::vec3 const& initial_target,
              glm::vec3 const& up);

  ~BasicCamera() = default;
  glm::mat4 GetTransform() override;
  void LookAt(glm::vec3 const& pos);
  void SetPos(glm::vec3 const& pos) override;
  void Move(glm::vec3 const& direction) override;
  void Rotate(glm::vec2 const& direction) override;

 private:
  glm::vec3 pos_, target_, up_;
};
}  // namespace Sculptor
