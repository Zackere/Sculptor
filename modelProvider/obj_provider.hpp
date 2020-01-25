#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "glm/glm.hpp"
#include "model_provider_base.hpp"

namespace Sculptor {
class ObjProvider : public ModelProviderBase {
 public:
  ObjProvider(std::string path) : path_(path) {}
  ~ObjProvider() override = default;

  bool Get(std::vector<glm::vec3>& out_vertices,
           std::vector<glm::vec2>& out_uvs,
           std::vector<glm::vec3>& out_normals) override;

 private:
  std::string path_;
};
}  // namespace Sculptor
