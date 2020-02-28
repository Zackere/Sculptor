// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <memory>

namespace Sculptor {
class glObject;
class glInstancedObject;

class SculptingMaterial {
 public:
  virtual ~SculptingMaterial();

  virtual void CollideWith(glObject& object) = 0;
  glInstancedObject& GetObject();

 protected:
  explicit SculptingMaterial(std::unique_ptr<glInstancedObject> tree);

 private:
  std::unique_ptr<glInstancedObject> tree_;
};
}  // namespace Sculptor
