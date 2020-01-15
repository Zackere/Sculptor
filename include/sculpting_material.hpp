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

  auto GetVerticiesBuffer() const { return reference_model_gl_.verticies; }
  auto GetUVBuffer() const { return reference_model_gl_.uvs; }
  auto GetNormalsBuffer() const { return reference_model_gl_.normals; }
  auto GetIndices() const { return reference_model_gl_.verticies; }
  auto GetNVertices() const { return reference_model_.verticies.size(); }
  auto const& GetMaterialElements() const { return offsets; }
  auto GetMaterialElementsBuffer() const { return offsets_buffer; }

  void RemoveAt(unsigned index);

 private:
  struct {
    std::vector<glm::vec3> verticies = {};
    std::vector<glm::vec2> uvs = {};
    std::vector<glm::vec3> normals = {};
  } reference_model_;
  struct {
    GLuint verticies = 0, uvs = 0, normals = 0;
  } reference_model_gl_;
  std::vector<glm::vec3> offsets = {};
  GLuint offsets_buffer = 0;
};
}  // namespace Sculptor
