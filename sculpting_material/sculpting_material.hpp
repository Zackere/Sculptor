#pragma once

#include <glm/glm.hpp>
#include <memory>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class MatrixApplierBase;
class ShapeGeneratorBase;
class glObject;
class glInstancedObject;

class SculptingMaterial {
 public:
  SculptingMaterial(int nobjects_start,
                    int nobjects_max,
                    std::unique_ptr<glObject> reference_model,
                    std::unique_ptr<ShapeGeneratorBase> outside_shape_generator,
                    std::unique_ptr<ShapeGeneratorBase> inside_shape_generator,
                    std::unique_ptr<MatrixApplierBase> matrix_applier);
  ~SculptingMaterial();

  void Render(glm::mat4 const& vp);
  void RotateLeft();
  void RotateRight();

 private:
  std::unique_ptr<glInstancedObject> visible_material_;
  CudaGraphicsResource<glm::vec3> invisible_material_;
};
}  // namespace Sculptor
