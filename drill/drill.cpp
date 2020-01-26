#include "drill.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>

#include "../glObject/gl_object.hpp"

namespace Sculptor {
Drill::Drill(std::unique_ptr<glObject> model) : model_(std::move(model)) {
  model_->Transform(
      glm::rotate(glm::translate(glm::mat4(1.f), glm::vec3(1.5, 0, 0)),
                  glm::pi<float>() / 2, glm::vec3(0, 0, -1)));

  forward_ = glm::vec3(-1, 0, 0);
}

Drill::~Drill() = default;

void Drill::Render(glm::mat4 const& vp) {
  model_->Render(vp);
}

void Drill::Spin() {
  auto avg_pos = model_->GetAvgPosition();
  model_->Transform(glm::translate(
      glm::rotate(glm::translate(glm::mat4(1.f), avg_pos), 0.1f, forward_),
      -avg_pos));
}

void Drill::MoveUp() {
  model_->Transform(glm::translate(glm::mat4(1.f), glm::vec3(0, 0.01f, 0)));
}

void Drill::MoveDown() {
  model_->Transform(glm::translate(glm::mat4(1.f), glm::vec3(0, -0.01f, 0)));
}

void Drill::MoveForward() {
  model_->Transform(glm::translate(glm::mat4(1.f), 0.01f * forward_));
}

void Drill::MoveBackward() {
  model_->Transform(glm::translate(glm::mat4(1.f), -0.01f * forward_));
}

}  // namespace Sculptor
