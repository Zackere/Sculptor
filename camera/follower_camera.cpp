// Copyright 2020 Wojciech Replin. All rights reserved.

#include "follower_camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include "../glObject/gl_object.hpp"
#include "basic_camera.hpp"

namespace Sculptor {

FollowerCamera::FollowerCamera(glm::vec3 const& initial_pos,
                               glObject* target,
                               glm::vec3 const& up)
    : target_(target),
      basic_camera_(
          new BasicCamera(initial_pos, target_->GetAvgPosition(), up)) {}

FollowerCamera::~FollowerCamera() = default;

glm::mat4 FollowerCamera::GetTransform() {
  Update();
  return basic_camera_->GetTransform();
}

void FollowerCamera::LookAt(glObject* object) {
  target_ = object;
}

glm::vec3 FollowerCamera::SetPos(glm::vec3 const& pos) {
  Update();
  return basic_camera_->SetPos(pos);
}

glm::vec3 FollowerCamera::Zoom(float amount) {
  Update();
  return basic_camera_->Zoom(amount);
}

glm::vec3 FollowerCamera::Rotate(glm::vec2 const& direction) {
  Update();
  return basic_camera_->Rotate(direction);
}

void FollowerCamera::Update() {
  basic_camera_->LookAt(target_->GetAvgPosition());
}

}  // namespace Sculptor
