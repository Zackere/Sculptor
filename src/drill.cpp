#include "../include/drill.hpp"

#include "glm/gtc/matrix_transform.hpp"

namespace Sculptor {
Drill::Drill()
    : glObject("../Sculptor/models/Drill.obj",
               "../Sculptor/shaders/DrillShader.vs",
               "../Sculptor/shaders/DrillShader.fs"),
      forward_(0, -1, 0) {
  model_matrix_ = glm::translate(model_matrix_, glm::vec3(2, 0, 0));
  model_matrix_ = glm::scale(model_matrix_, glm::vec3(0.03, 0.03, 0.03));
  model_matrix_ =
      glm::rotate(model_matrix_, glm::pi<float>() / 2, glm::vec3(0, 0, -1));
  glm::vec4 forward4(forward_, 1.0f);
  forward4 = glm::normalize(forward4);
  forward_ = forward4 / 10.f;
}

void Drill::MoveForward() {
  model_matrix_ = glm::translate(model_matrix_, forward_);
}

void Drill::MoveBackward() {
  model_matrix_ = glm::translate(model_matrix_, -forward_);
}

void Drill::NextFrame() {
  model_matrix_ = glm::rotate(model_matrix_, 0.01f, forward_);
}
}  // namespace Sculptor
