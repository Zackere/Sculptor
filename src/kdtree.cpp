#include "../include/kdtree.hpp"

#include "../include/drill.hpp"
#include "../include/sculpting_material.hpp"

namespace Sculptor {
void KdTree::Collide(Sculptor::SculptingMaterial* material,
                     Sculptor::Drill const& drill,
                     float tolerance) {
  for (auto i = 0u; i < material->GetMaterialElements().size(); ++i)
    for (auto const& v : drill.GetVertices())
      if (glm::length(v - material->GetMaterialElements()[i]) < tolerance)
        material->RemoveAt(i--);
}
}  // namespace Sculptor
