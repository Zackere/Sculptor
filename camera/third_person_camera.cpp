// Copyright 2020 Wojciech Replin. All rights reserved.

#include "third_person_camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include "../glObject/gl_object.hpp"
#include "basic_camera.hpp"

namespace Sculptor {

ThirdPersonCamera::ThirdPersonCamera(glm::vec3 const& initial_pos_offset,
                                     Sculptor::glObject* target,
                                     glm::vec3 const& up)
    : target_(target),
      basic_camera_(
          new BasicCamera(target_->GetAvgPosition() + initial_pos_offset,
                          target_->GetAvgPosition(),
                          up)),
      offset(initial_pos_offset) {}

glm::mat4 ThirdPersonCamera::GetTransform() {
  Update();
  return basic_camera_->GetTransform();
}

void ThirdPersonCamera::LookAt(glObject* object) {
  target_ = object;
}

glm::vec3 ThirdPersonCamera::SetPos(const glm::vec3& pos) {
  Update();
  glm::vec3 ret = basic_camera_->SetPos(pos);
  offset = ret - target_->GetAvgPosition();
  return ret;
}

glm::vec3 ThirdPersonCamera::Zoom(float amount) {
  Update();
  glm::vec3 ret = basic_camera_->Zoom(amount);
  offset = ret - target_->GetAvgPosition();
  return ret;
}

glm::vec3 ThirdPersonCamera::Rotate(const glm::vec2& direction) {
  Update();
  glm::vec3 ret = basic_camera_->Rotate(direction);
  offset = ret - target_->GetAvgPosition();
  return ret;
}

void ThirdPersonCamera::Update() {
  basic_camera_->SetPos(target_->GetAvgPosition() + offset);
  basic_camera_->LookAt(target_->GetAvgPosition());
}

ThirdPersonCamera::~ThirdPersonCamera() = default;

}  // namespace Sculptor
