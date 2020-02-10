// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "kdtree_constructor_base.hpp"

namespace Sculptor {
class KdTreeCPUStdConstructor : public KdTreeConstructor::Algorithm {
 public:
  ~KdTreeCPUStdConstructor() override = default;

  void Construct(float* x, float* y, float* z, int size) override;
};
}  // namespace Sculptor
