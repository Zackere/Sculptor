// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include "cube_sculpting_material.hpp"

namespace Sculptor {
class glObject;

class CollisionAlgorithmCPU : public CubeSculptingMaterial::CollisionAlgorithm {
 public:
  ~CollisionAlgorithmCPU() override = default;
  void Run(CubeSculptingMaterial* material, glObject& object) override;

 private:
  void FindNearestImpl(int depth, CubeSculptingMaterial::Node* root);

  CubeSculptingMaterial* material_ = nullptr;
  glm::vec3 query_point_ = {0, 0, 0};

  CubeSculptingMaterial::Node* current_nearest_node_ = nullptr;
  glm::vec3 current_nearest_ = {0, 0, 0};
  float best_distance_ = 0;
};
}  // namespace Sculptor
