// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "shape_generator_base.hpp"

namespace Sculptor {
class HollowCubeGenerator : public ShapeGeneratorBase {
 public:
  HollowCubeGenerator(float side_len) : side_len_(side_len) {}
  ~HollowCubeGenerator() override = default;
  std::vector<glm::vec3> Generate(int ncubes_on_side) override;
  int GetNumberOfOutputs(int ncubes_on_side) override {
    return ncubes_on_side * 6 * (ncubes_on_side - 2) + 8;
  }

 private:
  float side_len_ = 0;
};
}  // namespace Sculptor
