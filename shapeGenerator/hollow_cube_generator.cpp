// Copyright 2020 Wojciech Replin. All rights reserved.

#include "hollow_cube_generator.hpp"

namespace Sculptor {
std::vector<glm::vec3> HollowCubeGenerator::Generate(int ncubes_on_side) {
  std::vector<glm::vec3> ret;
  if (ncubes_on_side <= 0)
    return ret;
  if (ncubes_on_side == 1) {
    ret.emplace_back(0, 0, 0);
    return ret;
  }
  ret.reserve(GetNumberOfOutputs(ncubes_on_side));
  const auto start = -(ncubes_on_side - 3) * side_len_ / 2;
  float y, z;
  for (auto x : {start - side_len_, -start + side_len_}) {
    y = start;
    for (int j = 2; j < ncubes_on_side; ++j, y += side_len_) {
      z = start;
      for (int k = 2; k < ncubes_on_side; ++k, z += side_len_) {
        ret.emplace_back(x, y, z);
        ret.emplace_back(y, x, z);
        ret.emplace_back(y, z, x);
      }
      ret.emplace_back(y, x, x);
      ret.emplace_back(y, x, -x);
      ret.emplace_back(x, y, x);
      ret.emplace_back(x, y, -x);
      ret.emplace_back(x, x, y);
      ret.emplace_back(x, -x, y);
    }
    ret.emplace_back(x, x, x);
    ret.emplace_back(-x, x, x);
    ret.emplace_back(x, -x, x);
    ret.emplace_back(x, x, -x);
  }
  return ret;
}
}  // namespace Sculptor
