// Copyright 2020 Wojciech Replin. All rights reserved.

#include "basic_camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>
#undef GLM_ENABLE_EXPERIMENTAL
#else
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>
#endif

namespace Sculptor {
BasicCamera::BasicCamera(glm::vec3 const& initial_pos,
                         glm::vec3 const& initial_target,
                         glm::vec3 const& up)
    : pos_(initial_pos), target_(initial_target), up_(up) {}

glm::mat4 BasicCamera::GetTransform() {
  return glm::lookAt(pos_, target_, up_);
}

void BasicCamera::LookAt(glm::vec3 const& pos) {
  target_ = pos;
}

void BasicCamera::SetPos(glm::vec3 const& pos) {
  pos_ = pos;
}

void BasicCamera::Zoom(float amount) {
  pos_ += glm::normalize(target_ - pos_) * amount;
}

void BasicCamera::Rotate(glm::vec2 const& direction) {
  if (direction.x * direction.x + direction.y * direction.y < 0.0001f)
    return;
  pos_ -= target_;

  glm::vec3 local_right = glm::cross(pos_, up_);
  pos_ = glm::rotate(pos_, direction.y / 300, local_right);
  glm::vec3 local_up = glm::cross(pos_, local_right);
  pos_ = glm::rotate(pos_, direction.x / 300, local_up);

  pos_ += target_;
}
}  // namespace Sculptor
