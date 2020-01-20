#pragma once

#include <memory>
#include <vector>

#include "../include/kdtree.hpp"
#include "./glObject.hpp"
#include "GL/glew.h"
#include "glm/glm.hpp"

namespace Sculptor {
class Drill;

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
                    int size,
                    std::unique_ptr<KdTree> kd_tree);

  void Reset(InitialShape new_shape, int size);
  void Rotate(float amount);
  void Collide(Drill const& drill);

  void Render(glm::mat4 const& vp) const override;
  void Transform(glm::mat4 const& m) override;

 private:
  std::vector<glm::vec3> invisible_instances_positions_ = {};
  std::vector<glm::vec3> visible_instances_positions_ = {};
  GLuint visible_instances_positions_buffer_ = 0;
  GLuint texture_ = 0;
  std::unique_ptr<KdTree> kd_tree_ = nullptr;
  float side_len_ = 0;
  float angle_ = 0;
};
}  // namespace Sculptor
