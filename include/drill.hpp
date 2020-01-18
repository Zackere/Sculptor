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

 private:
  glm::vec3 forward_;
};
}  // namespace Sculptor
