// Copyright 2020 Wojciech Replin. All rights reserved.

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
class KdTreeConstructor;
class KdTreeRemover;

class CubeSculptingMaterial {
 public:
  CubeSculptingMaterial(
      int ncubes_per_side,
      std::unique_ptr<glObject> reference_model,
      std::unique_ptr<MatrixApplierBase> matrix_applier,
      std::unique_ptr<KdTreeConstructor> kd_tree_constructor,
      std::unique_ptr<KdTreeRemover> nearest_neighbour_finder);
  ~CubeSculptingMaterial();

  void Render(glm::mat4 const& vp);
  void Rotate(float amount);

  void Collide(glObject& object);

 private:
  float angle_ = 0;
  float side_len_;
  HollowCubeGenerator hollow_cube_generator_;
  CubeGenerator cube_generator_;
  std::unique_ptr<glInstancedObject> visible_material_;
  CudaGraphicsResource<float> invisible_material_x_, invisible_material_y_,
      invisible_material_z_;
  std::unique_ptr<KdTreeConstructor> kd_tree_constructor_;
  std::unique_ptr<KdTreeRemover> nearest_neighbour_finder_;
  std::unique_ptr<MatrixApplierBase> matrix_applier_;
};
}  // namespace Sculptor
