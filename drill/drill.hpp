#pragma once

#include <glm/glm.hpp>
#include <memory>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "../glObject/gl_object.hpp"

namespace Sculptor {
class MatrixApplierBase;
class Drill {
 public:
  Drill(std::unique_ptr<glObject> model);

  void Render(glm::mat4 const& vp) { model_->Render(vp); }

  void Spin();
  void MoveUp();
  void MoveDown();
  void MoveForward();
  void MoveBackward();

 private:
  std::unique_ptr<glObject> model_ = nullptr;
  glm::vec3 forward_ = {};
};
}  // namespace Sculptor
