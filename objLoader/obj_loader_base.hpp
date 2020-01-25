#pragma once

#include <string_view>
#include <vector>

#include "glm/glm.hpp"

namespace Sculptor {
class ObjLoaderBase {
 public:
  virtual ~ObjLoaderBase() = default;

  virtual bool Load(std::string_view path,
                    std::vector<glm::vec3>& out_vertices,
                    std::vector<glm::vec2>& out_uvs,
                    std::vector<glm::vec3>& out_normals) = 0;
};
}  // namespace Sculptor
