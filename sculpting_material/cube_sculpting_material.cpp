#include "cube_sculpting_material.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>

#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"

namespace Sculptor {
CubeSculptingMaterial::CubeSculptingMaterial(
    int ncubes_per_side,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : hollow_cube_generator_(2.f / ncubes_per_side),
      cube_generator_(
          std::make_unique<HollowCubeGenerator>(2.f / ncubes_per_side)),
      visible_material_(std::make_unique<glInstancedObject>(
          ncubes_per_side,
          hollow_cube_generator_.GetNumberOfOutputs(ncubes_per_side) +
              cube_generator_.GetNumberOfOutputs(ncubes_per_side - 2),
          std::move(reference_model),
          std::make_unique<HollowCubeGenerator>(2.f / ncubes_per_side),
          std::move(matrix_applier))),
      invisible_material_(
          cube_generator_.GetNumberOfOutputs(ncubes_per_side - 2)) {
  auto offsets = cube_generator_.Generate(ncubes_per_side);
  invisible_material_.SetData(offsets.data(), offsets.size());
}

CubeSculptingMaterial::~CubeSculptingMaterial() = default;

void CubeSculptingMaterial::Render(glm::mat4 const& vp) {
  visible_material_->Render(vp);
}

void CubeSculptingMaterial::RotateLeft() {
  visible_material_->Transform(
      glm::rotate(glm::mat4(1.f), -0.1f, glm::vec3(0, 1, 0)));
}

void CubeSculptingMaterial::RotateRight() {
  visible_material_->Transform(
      glm::rotate(glm::mat4(1.f), 0.1f, glm::vec3(0, 1, 0)));
}
}  // namespace Sculptor
