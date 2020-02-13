// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "kdtree_remover_base.hpp"

namespace Sculptor {
class KdTreeGPURemoverHeurestic : public KdTreeRemover::Algorithm {
 public:
  ~KdTreeGPURemoverHeurestic() override = default;

  std::vector<glm::vec3> RemoveNearest(float* x,
                                       float* y,
                                       float* z,
                                       int kd_size,
                                       float* query_points,
                                       int query_points_size,
                                       float threshold) override;
};
}  // namespace Sculptor
