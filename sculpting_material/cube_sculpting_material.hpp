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
class KdTree;

class CubeSculptingMaterial {
 public:
  CubeSculptingMaterial(int ncubes_per_side,
                        std::unique_ptr<glObject> reference_model,
                        std::unique_ptr<MatrixApplierBase> matrix_applier,
                        std::unique_ptr<KdTree> kd_tree_constructor,
                        std::unique_ptr<KdTree> nearest_neighbour_finder);
  ~CubeSculptingMaterial();

  void Render(glm::mat4 const& vp);
  void RotateLeft();
  void RotateRight();

  void Collide(glObject& object);

 private:
  float side_len_;
  HollowCubeGenerator hollow_cube_generator_;
  CubeGenerator cube_generator_;
  std::unique_ptr<glInstancedObject> visible_material_;
  CudaGraphicsResource<glm::vec3> invisible_material_;
  std::unique_ptr<KdTree> kd_tree_constructor_;
  std::unique_ptr<KdTree> nearest_neighbour_finder_;
};
}  // namespace Sculptor
