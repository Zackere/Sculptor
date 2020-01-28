#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "kdtree.hpp"

namespace Sculptor {
class KdTreeCPU : public KdTree {
 public:
  virtual ~KdTreeCPU() = default;

  void Construct(CudaGraphicsResource<float>& x,
                 CudaGraphicsResource<float>& y,
                 CudaGraphicsResource<float>& z) override;

  void RemoveNearest(CudaGraphicsResource<float>& kd_x,
                     CudaGraphicsResource<float>& kd_y,
                     CudaGraphicsResource<float>& kd_z,
                     CudaGraphicsResource<glm::vec3>& query_points,
                     float threshold) override;

 private:
  struct Zip {
    float *x, *y, *z;
  };
  glm::vec3 query_point_ = {0, 0, 0};
  std::vector<Zip>::iterator closest_node_ = std::vector<Zip>::iterator();
  float best_distance_squared_ = 0.f;

  float DistFromQuery(Zip const& zip) {
    return (*zip.x - query_point_.x) * (*zip.x - query_point_.x) +
           (*zip.y - query_point_.y) * (*zip.y - query_point_.y) +
           (*zip.z - query_point_.z) * (*zip.z - query_point_.z);
  }
  void FindNearestRecursive(std::vector<Zip>::iterator begin,
                            std::vector<Zip>::iterator end,
                            int level);
};
}  // namespace Sculptor
