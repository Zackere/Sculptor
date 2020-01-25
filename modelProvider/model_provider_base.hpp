#pragma once

#include <string_view>
#include <vector>

#include "glm/glm.hpp"

namespace Sculptor {
class ModelProviderBase {
 public:
  virtual ~ModelProviderBase() = default;

  virtual bool Get(std::vector<glm::vec3>& out_vertices,
                   std::vector<glm::vec2>& out_uvs,
                   std::vector<glm::vec3>& out_normals) = 0;
};
}  // namespace Sculptor
