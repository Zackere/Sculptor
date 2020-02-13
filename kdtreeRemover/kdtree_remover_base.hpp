// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class KdTreeRemover {
 public:
  class Algorithm {
   public:
    virtual ~Algorithm() = default;
    virtual std::vector<glm::vec3> RemoveNearest(float* x,
                                                 float* y,
                                                 float* z,
                                                 int kd_size,
                                                 float* query_points,
                                                 int query_points_size,
                                                 float threshold) = 0;
  };
  explicit KdTreeRemover(std::unique_ptr<Algorithm> alg)
      : remove_algorithm_(std::move(alg)) {}

  std::vector<glm::vec3> RemoveNearest(
      CudaGraphicsResource<float>& kd_x,
      CudaGraphicsResource<float>& kd_y,
      CudaGraphicsResource<float>& kd_z,
      CudaGraphicsResource<glm::vec3>& query_points,
      float threshold);

 private:
  std::unique_ptr<Algorithm> remove_algorithm_ = nullptr;
};
}  // namespace Sculptor
