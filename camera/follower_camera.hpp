// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <memory>

#include "camera_base.hpp"

namespace Sculptor {
class glObject;
class BasicCamera;

class FollowerCamera : public Camera {
 public:
  FollowerCamera(glm::vec3 const& initial_pos,
                 glObject* target,
                 glm::vec3 const& up);

  ~FollowerCamera();
  glm::mat4 GetTransform() override;
  void LookAt(glObject* object);
  void SetPos(glm::vec3 const& pos) override;
  void Zoom(float amount) override;
  void Rotate(glm::vec2 const& direction) override;

 private:
  glObject* target_ = nullptr;
  std::unique_ptr<BasicCamera> basic_camera_ = nullptr;

  FollowerCamera(FollowerCamera const&) = delete;
  FollowerCamera& operator=(FollowerCamera const&) = delete;
};
}  // namespace Sculptor
