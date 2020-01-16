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
  model_matrix_ *= glm::translate(glm::mat4(1.f), glm::vec3(2, 0, 0));
  model_matrix_ *= glm::scale(glm::mat4(1.f), glm::vec3(0.03, 0.03, 0.03));
  model_matrix_ *=
      glm::rotate(glm::mat4(1.f), glm::pi<float>() / 2, glm::vec3(0, 0, -1));
  MatrixApplier::Apply(reference_model_.verticies.data(),
                       reference_model_.verticies.size(), model_matrix_);
  glm::vec4 forward4(forward_, 1.0f);
  forward4 = glm::normalize(forward4) * model_matrix_;
  forward_ = forward4 / 10.f;
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.verticies);
  glBufferData(GL_ARRAY_BUFFER,
               reference_model_.verticies.size() * 3 * sizeof(float),
               reference_model_.verticies.data(), GL_STATIC_DRAW);
  model_matrix_ = glm::mat4(1.f);
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
  model_matrix_ = glm::rotate(model_matrix_, 0.01f, forward_);
  MatrixApplier::Apply(reference_model_.verticies.data(),
                       reference_model_.verticies.size(), model_matrix_);
  glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.verticies);
  glBufferData(GL_ARRAY_BUFFER,
               reference_model_.verticies.size() * 3 * sizeof(float),
               reference_model_.verticies.data(), GL_STATIC_DRAW);
  model_matrix_ = glm::mat4(1.f);
}
}  // namespace Sculptor
