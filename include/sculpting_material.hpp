#pragma once

#include <vector>

#include "./glObject.hpp"
#include "GL/glew.h"
#include "glm/glm.hpp"

namespace Sculptor {
class SculptingMaterial : public glObject {
 public:
  enum class InitialShape {
    CUBE,
  };
  enum class MaterialType {
    CUBE,
  };

  SculptingMaterial(MaterialType material_type,
                    InitialShape initial_shape,
                    int size);
  void Reset(InitialShape new_shape, int size);

  auto GetMaterialElements() const { return offsets_; }
  auto GetMaterialElementsBuffer() const { return offsets_buffer_; }
  auto GetTexture() const { return texture_; }

  void RemoveAt(unsigned index);
  void Rotate(float amount);

  void Enable() const override;
  void Render(glm::mat4 const& vp) const override;
  void Transform(glm::mat4 const& m) override;

 private:
  std::vector<glm::vec3> offsets_ = {};
  GLuint offsets_buffer_ = 0;
  GLuint texture_ = 0;
};
}  // namespace Sculptor
