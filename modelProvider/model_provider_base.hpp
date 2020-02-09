// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace Sculptor {
class ModelProviderBase {
 public:
  virtual ~ModelProviderBase() = default;

  virtual bool Get(std::vector<glm::vec3>& out_vertices,
                   std::vector<glm::vec2>& out_uvs,
                   std::vector<glm::vec3>& out_normals) = 0;
};
}  // namespace Sculptor
