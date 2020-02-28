// Copyright 2020 Wojciech Replin. All rights reserved.

#include "collision_algorithm_cpu.hpp"

#include <numeric>
#include <variant>

#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"

namespace Sculptor {
namespace {
constexpr float kEps = 0.001f;

float Len(glm::vec3 const& v) {
  return std::max(std::abs(v.x), std::max(std::abs(v.y), std::abs(v.z)));
}
}  // namespace
void CollisionAlgorithmCPU::Run(CubeSculptingMaterial* material,
                                glObject& object) {
  material_ = material;
  for (auto& p : object.GetVertices()->ToStdVector()) {
    query_point_ = glm::vec4(p, 1);
    current_nearest_ = {0, 0, 0};
    current_nearest_node_ = nullptr;
    best_distance_ = std::numeric_limits<float>::infinity();
    FindNearestImpl(material->GetMaxDepth(), tree_);
    if (material->ShouldRemoveNode(best_distance_))
      current_nearest_node_->Remove(material);
  }
  material_ = nullptr;
}

void CollisionAlgorithmCPU::FindNearestImpl(int depth,
                                            CubeSculptingMaterial::Node* root) {
  if (auto* i = std::get_if<unsigned>(&root->v)) {
    if (depth > 0) {
      root->Subdivide(material_);
    } else {
      auto m = material_->GetObject().GetTransformAt(*i);
      auto p = glm::vec3(m * glm::vec4{0, 0, 0, 1});
      auto dist = Len(query_point_ - p);
      if (best_distance_ > dist) {
        best_distance_ = dist;
        current_nearest_ = p;
        current_nearest_node_ = root;
      }
      return;
    }
  }
  auto diff = reinterpret_cast<float*>(&query_point_)[root->axis] -
              std::get<float>(root->v);
  if (diff >= -kEps) {
    if (root->r)
      FindNearestImpl(depth - 1, root->r.get());
    if (root->l && best_distance_ > diff)
      FindNearestImpl(depth - 1, root->l.get());
  }
  if (diff <= kEps) {
    if (root->l)
      FindNearestImpl(depth - 1, root->l.get());
    if (root->r && best_distance_ > -diff)
      FindNearestImpl(depth - 1, root->r.get());
  }
}
}  // namespace Sculptor
