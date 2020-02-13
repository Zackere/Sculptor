// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <memory>

#include "camera_base.hpp"

namespace Sculptor {
class glObject;
class BasicCamera;

class ThirdPersonCamera : public Camera {
 public:
  ThirdPersonCamera(glm::vec3 const& initial_pos_offset,
                    glObject* target,
                    glm::vec3 const& up);

  ~ThirdPersonCamera() override;
  glm::mat4 GetTransform() override;
  void LookAt(glObject* object);
  glm::vec3 SetPos(glm::vec3 const& pos) override;
  glm::vec3 Zoom(float amount) override;
  glm::vec3 Rotate(glm::vec2 const& direction) override;
  void LoadIntoShader(ShaderProgramBase* shader) override;

 private:
  void Update();

  glObject* target_ = nullptr;
  std::unique_ptr<BasicCamera> basic_camera_ = nullptr;
  glm::vec3 offset = {};

  ThirdPersonCamera(ThirdPersonCamera const&) = delete;
  ThirdPersonCamera& operator=(ThirdPersonCamera const&) = delete;
};
}  // namespace Sculptor
