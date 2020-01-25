#pragma once

#include <string_view>
#include <vector>

#include "glm/glm.hpp"
#include "obj_loader_base.hpp"

namespace Sculptor {
class ObjLoader : public ObjLoaderBase {
 public:
  ~ObjLoader() override = default;

  bool Load(std::string_view path,
            std::vector<glm::vec3>& out_vertices,
            std::vector<glm::vec2>& out_uvs,
            std::vector<glm::vec3>& out_normals) override;
};
}  // namespace Sculptor
