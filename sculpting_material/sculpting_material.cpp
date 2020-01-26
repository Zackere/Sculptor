#include "sculpting_material.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>

#include "../glObject/gl_object.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"

namespace Sculptor {
SculptingMaterial::SculptingMaterial(
    int nobjects_visible,
    int nobjects_invisible,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<ShapeGeneratorBase> outside_shape_generator,
    std::unique_ptr<ShapeGeneratorBase> inside_shape_generator,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : visible_material_(nullptr),
      invisible_material_(
          inside_shape_generator->GetNumberOfOutputs(nobjects_invisible)) {
  int out_cap = inside_shape_generator->GetNumberOfOutputs(nobjects_invisible) +
                outside_shape_generator->GetNumberOfOutputs(nobjects_visible);

  visible_material_ = std::make_unique<glInstancedObject>(
      nobjects_visible, out_cap, std::move(reference_model),
      std::move(outside_shape_generator), std::move(matrix_applier));

  auto offsets = inside_shape_generator->Generate(nobjects_invisible);
  invisible_material_.SetData(offsets.data(), offsets.size());
}

void SculptingMaterial::Render(glm::mat4 const& vp) {
  visible_material_->Render(vp);
}

void SculptingMaterial::RotateLeft() {
  visible_material_->Transform(
      glm::rotate(glm::mat4(1.f), -0.1f, glm::vec3(0, 1, 0)));
}

void SculptingMaterial::RotateRight() {
  visible_material_->Transform(
      glm::rotate(glm::mat4(1.f), 0.1f, glm::vec3(0, 1, 0)));
}
}  // namespace Sculptor
