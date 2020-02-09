// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <memory>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class glObject;

class Drill {
 public:
  Drill(std::unique_ptr<glObject> model);
  ~Drill();

  void Render(glm::mat4 const& vp);

  void Spin();
  void MoveUp();
  void MoveDown();
  void MoveForward();
  void MoveBackward();

  glObject& GetObject() { return *model_; }

 private:
  std::unique_ptr<glObject> model_;
  glm::vec3 forward_ = {};
};
}  // namespace Sculptor
