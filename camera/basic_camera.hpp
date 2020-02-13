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
  glm::vec3 SetPos(glm::vec3 const& pos) override;
  glm::vec3 Zoom(float amount) override;
  glm::vec3 Rotate(glm::vec2 const& direction) override;
  void LoadIntoShader(ShaderProgramBase* shader) override;

 private:
  glm::vec3 pos_, target_, up_;
};
}  // namespace Sculptor
