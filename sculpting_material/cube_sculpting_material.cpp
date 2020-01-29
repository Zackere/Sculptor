#include "cube_sculpting_material.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>

#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"
#include "../kdtree/kdtree.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"

namespace Sculptor {
namespace {
constexpr float kSmallAngle = 0.01f;
}
CubeSculptingMaterial::CubeSculptingMaterial(
    int ncubes_per_side,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<MatrixApplierBase> matrix_applier,
    std::unique_ptr<KdTree> kd_tree_constructor,
    std::unique_ptr<KdTree> nearest_neighbour_finder)
    : side_len_(2.f / ncubes_per_side),
      hollow_cube_generator_(side_len_),
      cube_generator_(std::make_unique<HollowCubeGenerator>(side_len_)),
      visible_material_(std::make_unique<glInstancedObject>(
          ncubes_per_side,
          hollow_cube_generator_.GetNumberOfOutputs(ncubes_per_side) +
              cube_generator_.GetNumberOfOutputs(ncubes_per_side - 2),
          std::move(reference_model),
          std::make_unique<HollowCubeGenerator>(side_len_),
          std::move(matrix_applier))),
      invisible_material_(
          cube_generator_.GetNumberOfOutputs(ncubes_per_side - 2)),
      kd_tree_constructor_(std::move(kd_tree_constructor)),
      nearest_neighbour_finder_(std::move(nearest_neighbour_finder)) {
  auto offsets = cube_generator_.Generate(ncubes_per_side);
  invisible_material_.SetData(offsets.data(), offsets.size());
}

CubeSculptingMaterial::~CubeSculptingMaterial() {}

void CubeSculptingMaterial::Render(glm::mat4 const& vp) {
  visible_material_->Render(vp);
}

void CubeSculptingMaterial::RotateLeft() {
  visible_material_->Transform(
      glm::rotate(glm::mat4(1.f), -kSmallAngle, glm::vec3(0, 1, 0)));
}

void CubeSculptingMaterial::RotateRight() {
  visible_material_->Transform(
      glm::rotate(glm::mat4(1.f), kSmallAngle, glm::vec3(0, 1, 0)));
}

void CubeSculptingMaterial::Collide(glObject& object) {
  if (kd_tree_constructor_) {
    kd_tree_constructor_->Construct(visible_material_->GetVecticesX(),
                                    visible_material_->GetVecticesY(),
                                    visible_material_->GetVecticesZ());
    nearest_neighbour_finder_->RemoveNearest(
        visible_material_->GetVecticesX(), visible_material_->GetVecticesY(),
        visible_material_->GetVecticesZ(), *object.GetVertices(), side_len_ / 2,
        false);
  } else {
    nearest_neighbour_finder_->RemoveNearest(
        visible_material_->GetVecticesX(), visible_material_->GetVecticesY(),
        visible_material_->GetVecticesZ(), *object.GetVertices(), side_len_ / 2,
        true);
  }
}
}  // namespace Sculptor
