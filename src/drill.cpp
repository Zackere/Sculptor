#include "../include/drill.hpp"

#include "../include/objloader.hpp"
#include "../include/shader_loader.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Sculptor {
Drill::Drill()
    : glObject("../Sculptor/models/Drill.obj",
               "../Sculptor/shaders/DrillShader.vs",
               "../Sculptor/shaders/DrillShader.fs"),
      forward_(-1, 0, 0) {
  model_matrix_ = glm::scale(model_matrix_, glm::vec3(0.1, 0.1, 0.1));
  model_matrix_ = glm::translate(model_matrix_, glm::vec3(20, 0, 0));
  const auto rotation_axis = glm::vec3(0, 0, 1);
  model_matrix_ =
      glm::rotate(model_matrix_, -glm::pi<float>() / 2, rotation_axis);
  glm::vec4 forward4(forward_, 1.0f);
  forward4 = glm::rotate(glm::mat4(1.f), glm::pi<float>() / 2, rotation_axis) *
             forward4;
  forward4 = glm::normalize(forward4);
  forward_ = forward4 / 100.f;
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
