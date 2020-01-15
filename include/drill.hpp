#pragma once

#include <vector>

#include "../include/glObject.hpp"
#include "glm/glm.hpp"

namespace Sculptor {
class Drill : public glObject {
 public:
  Drill();

  void NextFrame();
  void MoveBackward();
  void MoveForward();
  auto const& GetVertices() const { return reference_model_.verticies; }

 private:
  glm::vec3 forward_;
};
}  // namespace Sculptor
