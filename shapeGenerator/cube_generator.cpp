#include "cube_generator.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

namespace Sculptor {

CubeGenerator::CubeGenerator(
    std::unique_ptr<HollowCubeGenerator> hollow_cube_gen)
    : hollow_cube_gen_(std::move(hollow_cube_gen)) {}

std::vector<glm::vec3> CubeGenerator::Generate(int ncubes_on_side) {
  std::vector<glm::vec3> ret;
  ret.reserve(ncubes_on_side * ncubes_on_side * ncubes_on_side);
  for (int i = ncubes_on_side; i > 0; i -= 2) {
    auto hollow = hollow_cube_gen_->Generate(i);
    std::copy(hollow.begin(), hollow.end(), std::back_inserter(ret));
  }
  return ret;
}

}  // namespace Sculptor
