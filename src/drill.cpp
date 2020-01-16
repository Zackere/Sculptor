#include "../include/drill.hpp"

#include <stdio.h>

#include <iostream>

#include "../include/matrix_applier.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Sculptor {
Drill::Drill()
    : glObject("../Sculptor/models/Drill.obj",
               "../Sculptor/shaders/DrillShader.vs",
               "../Sculptor/shaders/DrillShader.fs"),
      forward_(0, 1, 0) {
  auto model_matrix = glm::mat4(1.f);
  model_matrix *= glm::translate(glm::mat4(1.f), glm::vec3(2, 0, 0));
  model_matrix *= glm::scale(glm::mat4(1.f), glm::vec3(0.03, 0.03, 0.03));
  model_matrix *=
      glm::rotate(glm::mat4(1.f), glm::pi<float>() / 2, glm::vec3(0, 0, -1));
  MatrixApplier::Apply(reference_model_.verticies.data(),
                       reference_model_.verticies.size(), model_matrix);
  glm::vec4 forward4(forward_, 1.0f);
  forward4 = glm::normalize(forward4) * model_matrix;
  forward_ = forward4 / 10.f;
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.verticies);
  glBufferData(GL_ARRAY_BUFFER,
               reference_model_.verticies.size() * 3 * sizeof(float),
               reference_model_.verticies.data(), GL_STATIC_DRAW);
}

void Drill::MoveForward() {
  MatrixApplier::Apply(reference_model_.verticies.data(),
                       reference_model_.verticies.size(),
                       glm::translate(glm::mat4(1.f), forward_));
}

void Drill::MoveBackward() {
  MatrixApplier::Apply(reference_model_.verticies.data(),
                       reference_model_.verticies.size(),
                       glm::translate(glm::mat4(1.f), -forward_));
}

void Drill::NextFrame() {
  MatrixApplier::Apply(reference_model_.verticies.data(),
                       reference_model_.verticies.size(),
                       glm::rotate(glm::mat4(1.f), 0.01f, forward_));
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.verticies);
  glBufferData(GL_ARRAY_BUFFER,
               reference_model_.verticies.size() * 3 * sizeof(float),
               reference_model_.verticies.data(), GL_STATIC_DRAW);
}
}  // namespace Sculptor
