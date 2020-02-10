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
  basic_camera_->LookAt(target_->GetAvgPosition());
  return basic_camera_->GetTransform();
}

void FollowerCamera::LookAt(glObject* object) {
  target_ = object;
}

void FollowerCamera::SetPos(glm::vec3 const& pos) {
  basic_camera_->SetPos(pos);
}

void FollowerCamera::Move(glm::vec3 const& direction) {
  basic_camera_->Move(direction);
}

void FollowerCamera::Rotate(glm::vec2 const& direction) {
  basic_camera_->LookAt(target_->GetAvgPosition());
  basic_camera_->Rotate(direction);
}

}  // namespace Sculptor
