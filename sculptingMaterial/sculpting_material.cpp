// Copyright 2020 Wojciech Replin. All rights reserved.

#include "sculpting_material.hpp"

#include <utility>

#include "../glInstancedObject/gl_instanced_object.hpp"

namespace Sculptor {
SculptingMaterial::~SculptingMaterial() = default;

glInstancedObject& SculptingMaterial::GetObject() {
  return *tree_;
}

SculptingMaterial::SculptingMaterial(std::unique_ptr<glInstancedObject> tree)
    : tree_(std::move(tree)) {}
}  // namespace Sculptor
