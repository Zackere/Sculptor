#pragma once

#include <vector>

#include "GL/glew.h"
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

  auto GetVerticiesBuffer() const { return material_.verticies; }
  auto GetUVBuffer() const { return material_.uvs; }
  auto GetNormalsBuffer() const { return material_.normals; }
  auto GetNVerticies() const { return nverticies_; }
  auto const& GetMaterialElements() const { return material_.offsets; }

  void RemoveAt(unsigned index);

 private:
  struct {
    std::vector<glm::vec3> verticies = {};
    std::vector<glm::vec2> uvs = {};
    std::vector<glm::vec3> normals = {};
  } reference_model_;
  struct {
    GLuint verticies = 0, uvs = 0, normals = 0;
    std::vector<glm::vec3> offsets = {};
  } material_;
  GLsizei nverticies_ = 0;
};
}  // namespace Sculptor
