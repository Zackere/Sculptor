#include "cube_sculpting_material.hpp"

#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <utility>
#include <vector>

#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"
#include "../kdtree/kdtree.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"

namespace Sculptor {
namespace {
constexpr float kEps = 0.01f;
}
CubeSculptingMaterial::CubeSculptingMaterial(
    int ncubes_per_side,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<MatrixApplierBase> matrix_applier,
    std::unique_ptr<KdTreeConstructor> kd_tree_constructor,
    std::unique_ptr<KdTreeRemover> nearest_neighbour_finder)
    : side_len_(2.f / ncubes_per_side),
      hollow_cube_generator_(side_len_),
      cube_generator_(std::make_unique<HollowCubeGenerator>(side_len_)),
      visible_material_(std::make_unique<glInstancedObject>(
          ncubes_per_side,
          hollow_cube_generator_.GetNumberOfOutputs(ncubes_per_side) +
              cube_generator_.GetNumberOfOutputs(ncubes_per_side - 2),
          std::move(reference_model),
          std::make_unique<HollowCubeGenerator>(side_len_),
          matrix_applier->Clone())),
      invisible_material_x_(
          cube_generator_.GetNumberOfOutputs(ncubes_per_side - 2)),
      invisible_material_y_(
          cube_generator_.GetNumberOfOutputs(ncubes_per_side - 2)),
      invisible_material_z_(
          cube_generator_.GetNumberOfOutputs(ncubes_per_side - 2)),
      kd_tree_constructor_(std::move(kd_tree_constructor)),
      nearest_neighbour_finder_(std::move(nearest_neighbour_finder)),
      matrix_applier_(matrix_applier->Clone()) {
  auto off = cube_generator_.Generate(ncubes_per_side - 2);
  std::vector<float> coordinate(off.size());
  for (auto i = 0u; i < off.size(); ++i)
    coordinate[i] = off[i].x;
  invisible_material_x_.SetData(coordinate.data(), coordinate.size());
  for (auto i = 0u; i < off.size(); ++i)
    coordinate[i] = off[i].y;
  invisible_material_y_.SetData(coordinate.data(), coordinate.size());
  for (auto i = 0u; i < off.size(); ++i)
    coordinate[i] = off[i].z;
  invisible_material_z_.SetData(coordinate.data(), coordinate.size());
}

CubeSculptingMaterial::~CubeSculptingMaterial() {}

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

void CubeSculptingMaterial::Collide(glObject& object) {
  auto invisible_res = kd_tree_constructor_->LoadResources(
      invisible_material_x_, invisible_material_y_, invisible_material_z_);
  auto constructor_future = kd_tree_constructor_->Construct(
      std::get<0>(invisible_res), std::get<1>(invisible_res),
      std::get<2>(invisible_res), invisible_material_x_.GetSize());

  auto visible_res = kd_tree_constructor_->LoadResources(
      visible_material_->GetVecticesX(), visible_material_->GetVecticesY(),
      visible_material_->GetVecticesZ());
  kd_tree_constructor_
      ->Construct(std::get<0>(visible_res), std::get<1>(visible_res),
                  std::get<2>(visible_res),
                  visible_material_->GetVecticesX().GetSize())
      .get();
  kd_tree_constructor_->UnloadResources(std::get<0>(visible_res),
                                        std::get<1>(visible_res),
                                        std::get<2>(visible_res));

  std::vector<glm::vec3> removed = nearest_neighbour_finder_->RemoveNearest(
      visible_material_->GetVecticesX(), visible_material_->GetVecticesY(),
      visible_material_->GetVecticesZ(), *object.GetVertices(), side_len_ / 2);

  if (removed.empty()) {
    constructor_future.get();
    kd_tree_constructor_->UnloadResources(std::get<0>(invisible_res),
                                          std::get<1>(invisible_res),
                                          std::get<2>(invisible_res));
    return;
  }

  auto m_back = glm::rotate(glm::mat4(1.f), -angle_, glm::vec3{0, 1, 0});
  auto m_for = glm::rotate(glm::mat4(1.f), angle_, glm::vec3{0, 1, 0});

  std::vector<glm::vec3> to_be_added;
  to_be_added.reserve(removed.size() * 6);
  for (auto const& v : removed) {
    auto p = m_back * glm::vec4(v, 1.f);
    to_be_added.emplace_back(glm::vec3{p.x + side_len_, p.y, p.z});
    to_be_added.emplace_back(glm::vec3{p.x - side_len_, p.y, p.z});
    to_be_added.emplace_back(glm::vec3{p.x, p.y + side_len_, p.z});
    to_be_added.emplace_back(glm::vec3{p.x, p.y - side_len_, p.z});
    to_be_added.emplace_back(glm::vec3{p.x, p.y, p.z + side_len_});
    to_be_added.emplace_back(glm::vec3{p.x, p.y, p.z - side_len_});
  }
  CudaGraphicsResource<glm::vec3> to_be_added_resource(to_be_added.size());
  to_be_added_resource.SetData(to_be_added.data(), to_be_added.size());

  constructor_future.get();
  kd_tree_constructor_->UnloadResources(std::get<0>(invisible_res),
                                        std::get<1>(invisible_res),
                                        std::get<2>(invisible_res));

  removed = nearest_neighbour_finder_->RemoveNearest(
      invisible_material_x_, invisible_material_y_, invisible_material_z_,
      to_be_added_resource, kEps);

  matrix_applier_->Apply(removed, m_for);
  visible_material_->AddInstances(removed);
}
}  // namespace Sculptor
