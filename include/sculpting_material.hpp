#pragma once

#include <vector>

#include "glm/glm.hpp"

namespace Sculptor {
class SculptingMaterial {
 public:
  enum class InitialShape {
    CUBE,
  };
  enum class MaterialType {
    CUBE,
    SPHERE,
  };

  SculptingMaterial(MaterialType material_type,
                    InitialShape initial_shape,
                    int size);
  void Reset(InitialShape new_shape, int size);
  std::vector<glm::vec3> const& GetVerticiesProperty() const;
  std::vector<glm::vec2> const& GetUVSProperty() const;
  std::vector<glm::vec3> const& GetNormalsProperty() const;

 private:
  struct {
    std::vector<glm::vec3> verticies = {};
    std::vector<glm::vec2> uvs = {};
    std::vector<glm::vec3> normals = {};
  } reference_model_;
  struct {
    std::vector<glm::vec3> verticies = {};
    std::vector<glm::vec2> uvs = {};
    std::vector<glm::vec3> normals = {};
    std::vector<glm::vec3> offsets = {};
  } material_;
};
}  // namespace Sculptor
