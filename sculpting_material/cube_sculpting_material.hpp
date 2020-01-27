#pragma once

#include <glm/glm.hpp>
#include <memory>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "../shapeGenerator/cube_generator.hpp"
#include "../shapeGenerator/hollow_cube_generator.hpp"

namespace Sculptor {
class MatrixApplierBase;
class glObject;
class glInstancedObject;

class CubeSculptingMaterial {
 public:
  CubeSculptingMaterial(int ncubes_per_side,
                        std::unique_ptr<glObject> reference_model,
                        std::unique_ptr<MatrixApplierBase> matrix_applier);
  ~CubeSculptingMaterial();

  void Render(glm::mat4 const& vp);
  void RotateLeft();
  void RotateRight();

 private:
  HollowCubeGenerator hollow_cube_generator_;
  CubeGenerator cube_generator_;
  std::unique_ptr<glInstancedObject> visible_material_;
  CudaGraphicsResource<glm::vec3> invisible_material_;
};
}  // namespace Sculptor
