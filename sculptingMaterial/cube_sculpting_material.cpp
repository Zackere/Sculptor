// Copyright 2020 Wojciech Replin. All rights reserved.

#include "cube_sculpting_material.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>
#include <vector>

#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"
#include "../kdtreeConstructor/kdtree_constructor_base.hpp"
#include "../kdtreeRemover/kdtree_remover_base.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"

namespace Sculptor {
CubeSculptingMaterial::CubeSculptingMaterial(
    int ncubes_per_side,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : side_len_(2.f / ncubes_per_side),
      visible_material_(std::make_unique<glInstancedObject>(
          0,
          ncubes_per_side * ncubes_per_side * ncubes_per_side,
          std::move(reference_model),
          std::make_unique<HollowCubeGenerator>(side_len_),
          matrix_applier->Clone())),
      material_x_(ncubes_per_side * ncubes_per_side * ncubes_per_side),
      material_y_(ncubes_per_side * ncubes_per_side * ncubes_per_side),
      material_z_(ncubes_per_side * ncubes_per_side * ncubes_per_side),
      matrix_applier_(matrix_applier->Clone()) {
  material_x_.PushBack(0);
  material_y_.PushBack(0);
  material_z_.PushBack(0);
  visible_material_->AddInstances({glm::vec3{0, 0, 0}});
}

CubeSculptingMaterial::~CubeSculptingMaterial() = default;

void CubeSculptingMaterial::Render(glm::mat4 const& vp) {
  visible_material_->Render(vp);
}

void CubeSculptingMaterial::Rotate(float amount) {
  auto m = glm::rotate(glm::mat4(1.f), amount, glm::vec3(0, 1, 0));
  visible_material_->Transform(m);
  angle_ += amount;
  while (angle_ >= glm::two_pi<float>())
    angle_ -= glm::two_pi<float>();
  while (angle_ < 0)
    angle_ += glm::two_pi<float>();
}

void CubeSculptingMaterial::Collide(glObject&) {}

glInstancedObject& CubeSculptingMaterial::GetObject() {
  return *visible_material_;
}
}  // namespace Sculptor
