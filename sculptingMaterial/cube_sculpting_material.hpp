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
  CubeSculptingMaterial(int ncubes_per_side,
                        std::unique_ptr<glObject> reference_model,
                        std::unique_ptr<MatrixApplierBase> matrix_applier);
  ~CubeSculptingMaterial();

  void Render(glm::mat4 const& vp);
  void Rotate(float amount);

  void Collide(glObject& object);

  glInstancedObject& GetObject();

 private:
  float angle_ = 0;
  float side_len_;
  std::unique_ptr<glInstancedObject> visible_material_;
  CudaGraphicsResource<float> material_x_, material_y_, material_z_;
  std::unique_ptr<MatrixApplierBase> matrix_applier_;
};
}  // namespace Sculptor
