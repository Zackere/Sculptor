// Copyright 2020 Wojciech Replin. All rights reserved.

#include "kdtree_cpu_std_constructor.hpp"

#include <cuda.h>

#include <algorithm>
#include <execution>

namespace Sculptor {
namespace {
template <typename RandomIt>
void ConstructRecursive(RandomIt begin, RandomIt end, int level) {
  if (end <= begin)
    return;
  std::sort(std::execution::par, begin, end,
            [level](auto const& v1, auto const& v2) {
              return reinterpret_cast<float const*>(&v1)[level] <
                     reinterpret_cast<float const*>(&v2)[level];
            });
  level = level == 2 ? 0 : (level + 1);
  auto mid = begin + (end - begin) / 2;
  ConstructRecursive(begin, mid, level);
  ConstructRecursive(mid + 1, end, level);
}
}  // namespace
void KdTreeCPUStdConstructor::Construct(float* x,
                                        float* y,
                                        float* z,
                                        int size) {
  std::vector<float> kd_x(size), kd_y(size), kd_z(size);
  SculptorCudaCheckError(
      cudaMemcpy(kd_x.data(), x, size * sizeof(float), cudaMemcpyDeviceToHost));
  SculptorCudaCheckError(
      cudaMemcpy(kd_y.data(), y, size * sizeof(float), cudaMemcpyDeviceToHost));
  SculptorCudaCheckError(
      cudaMemcpy(kd_z.data(), z, size * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<glm::vec3> kd(size);
  for (auto i = 0u; i < kd.size(); ++i)
    kd[i] = {kd_x[i], kd_y[i], kd_z[i]};
  ConstructRecursive(kd.begin(), kd.end(), 0);
  for (auto i = 0u; i < kd.size(); ++i) {
    kd_x[i] = kd[i].x;
    kd_y[i] = kd[i].y;
    kd_z[i] = kd[i].z;
  }

  SculptorCudaCheckError(
      cudaMemcpy(x, kd_x.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  SculptorCudaCheckError(
      cudaMemcpy(y, kd_y.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  SculptorCudaCheckError(
      cudaMemcpy(z, kd_z.data(), size * sizeof(float), cudaMemcpyHostToDevice));
}
}  // namespace Sculptor
