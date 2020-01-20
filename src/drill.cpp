#include "../include/drill.hpp"

#include <stdio.h>

#include <iostream>

#include "glm/gtc/matrix_transform.hpp"

namespace Sculptor {
Drill::Drill()
    : glObject("../Sculptor/models/Drill.obj",
               "../Sculptor/shaders/DrillShader.vs",
               "../Sculptor/shaders/DrillShader.fs"),
      forward_(0, 1, 0) {
  auto model_matrix = glm::mat4(1.f);
  model_matrix = glm::translate(model_matrix, glm::vec3(2, 0, 0));
  model_matrix = glm::scale(model_matrix, glm::vec3(0.03, 0.03, 0.03));
  model_matrix =
      glm::rotate(model_matrix, glm::pi<float>() / 2, glm::vec3(0, 0, -1));
  Transform(model_matrix);
  glm::vec4 forward4(forward_, 1.0f);
  forward4 = glm::normalize(forward4) * model_matrix;
  forward_ = forward4;
}

void Drill::MoveForward() {
  Transform(glm::translate(glm::mat4(1.f), forward_));
}

void Drill::MoveBackward() {
  Transform(glm::translate(glm::mat4(1.f), -forward_));
}

void Drill::NextFrame() {
  Transform(glm::rotate(glm::mat4(1.f), 1.f, forward_));
}
}  // namespace Sculptor
