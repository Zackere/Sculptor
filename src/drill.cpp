#include "../include/drill.hpp"

#include <execution>
#include <iostream>
#include <numeric>

#include "glm/gtc/matrix_transform.hpp"
namespace Sculptor {
namespace {
constexpr float kScale = 0.03;
}
Drill::Drill()
    : glObject("../Sculptor/models/Drill.obj",
               "../Sculptor/shaders/DrillShader.vs",
               "../Sculptor/shaders/DrillShader.fs"),
      forward_(0, 1, 0),
      pos_(0, 0, 0) {
  auto model_matrix = glm::mat4(1.f);
  model_matrix = glm::translate(model_matrix, glm::vec3(2, 0, 0));
  model_matrix = glm::scale(model_matrix, glm::vec3(kScale, kScale, kScale));
  model_matrix =
      glm::rotate(model_matrix, glm::pi<float>() / 2, glm::vec3(0, 0, -1));
  Transform(model_matrix);
  glm::vec4 forward4(forward_, 1.0f);
  forward4 = glm::normalize(forward4) * model_matrix;
  forward_ = forward4;
  pos_ = std::reduce(std::execution::par, GetReferenceModelVertices().begin(),
                     GetReferenceModelVertices().end()) /
         static_cast<float>(GetReferenceModelVertices().size());
}

void Drill::MoveForward() {
  Transform(glm::translate(glm::mat4(1.f), forward_));
}

void Drill::MoveUp() {
  pos_ += glm::vec3(0, kScale, 0);
  Transform(glm::translate(glm::mat4(1.f), glm::vec3(0, kScale, 0)));
}

void Drill::MoveDown() {
  pos_ += glm::vec3(0, -kScale, 0);
  Transform(glm::translate(glm::mat4(1.f), glm::vec3(0, -kScale, 0)));
}

void Drill::MoveBackward() {
  Transform(glm::translate(glm::mat4(1.f), -forward_));
}

void Drill::NextFrame() {
  Transform(glm::translate(glm::mat4(1.f), -pos_));
  Transform(glm::rotate(glm::mat4(1.f), 1.f, forward_));
  Transform(glm::translate(glm::mat4(1.f), pos_));
}
}  // namespace Sculptor
