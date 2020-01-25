#pragma once

#include <vector>

#include "glm/glm.hpp"
#include "shape_generator_base.hpp"

namespace Sculptor {
class HollowCubeGenerator : public ShapeGeneratorBase {
 public:
  HollowCubeGenerator(float side_len) : side_len_(side_len) {}
  ~HollowCubeGenerator() override = default;
  std::vector<glm::vec3> Generate(int ncubes_on_side) override;

 private:
  float side_len_;
};
}  // namespace Sculptor
