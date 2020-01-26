#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "hollow_cube_generator.hpp"
#include "shape_generator_base.hpp"

namespace Sculptor {
class CubeGenerator : public ShapeGeneratorBase {
 public:
  CubeGenerator(std::unique_ptr<HollowCubeGenerator> hollow_cube_gen);
  ~CubeGenerator() override = default;
  std::vector<glm::vec3> Generate(int ncubes_on_side) override;
  int GetNumberOfOutputs(int ncubes_on_side) override {
    return ncubes_on_side * ncubes_on_side * ncubes_on_side;
  }

 private:
  std::unique_ptr<HollowCubeGenerator> hollow_cube_gen_ = nullptr;
};
}  // namespace Sculptor
