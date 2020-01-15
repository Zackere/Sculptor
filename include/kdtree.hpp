#pragma once

namespace Sculptor {
class SculptingMaterial;
class Drill;
class KdTree {
 public:
  virtual ~KdTree() = default;

  virtual void Collide(SculptingMaterial* material,
                       Drill const& drill,
                       float tolerance);
};
}  // namespace Sculptor
