#include "kdtree_cpu.hpp"

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <execution>
#include <functional>
#include <set>
#include <utility>

#include "../util/cudaCheckError.hpp"

namespace Sculptor {
namespace {
constexpr float kEps = 0.001f;

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
void KdTreeCPU::Construct(float* x, float* y, float* z, int size) {
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

std::vector<glm::vec3> KdTreeCPU::RemoveNearest(float* x,
                                                float* y,
                                                float* z,
                                                int kd_size,
                                                float* query_points,
                                                int query_points_size,
                                                float threshold) {
  std::vector<float> kd_x(kd_size), kd_y(kd_size), kd_z(kd_size);
  SculptorCudaCheckError(cudaMemcpy(kd_x.data(), x, kd_size * sizeof(float),
                                    cudaMemcpyDeviceToHost));
  SculptorCudaCheckError(cudaMemcpy(kd_y.data(), y, kd_size * sizeof(float),
                                    cudaMemcpyDeviceToHost));
  SculptorCudaCheckError(cudaMemcpy(kd_z.data(), z, kd_size * sizeof(float),
                                    cudaMemcpyDeviceToHost));
  std::vector<glm::vec3> queries(query_points_size);
  SculptorCudaCheckError(cudaMemcpy(queries.data(), query_points,
                                    query_points_size * sizeof(glm::vec3),
                                    cudaMemcpyDeviceToHost));

  std::vector<glm::vec3> kd(kd_size);
  for (auto i = 0u; i < kd.size(); ++i)
    kd[i] = {kd_x[i], kd_y[i], kd_z[i]};

  std::set<std::vector<glm::vec3>::iterator, std::greater<>> to_be_removed;
  for (auto const& v : queries) {
    closest_node_ = kd.begin() + (kd.end() - kd.begin()) / 2;
    query_point_ = v;
    best_distance_ = DistFromQuery(*closest_node_);
    FindNearestRecursive(kd.begin(), kd.end(), 0);
    if (best_distance_ < threshold)
      to_be_removed.insert(closest_node_);
  }
  std::vector<glm::vec3> ret;
  ret.reserve(to_be_removed.size());
  for (auto it : to_be_removed) {
    ret.emplace_back(*it);
    using std::swap;
    swap(*it, kd.back());
    kd.pop_back();
  }
  for (auto i = 0u; i < kd.size(); ++i) {
    kd_x[i] = kd[i].x;
    kd_y[i] = kd[i].y;
    kd_z[i] = kd[i].z;
  }

  SculptorCudaCheckError(cudaMemcpy(x, kd_x.data(), kd_x.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
  SculptorCudaCheckError(cudaMemcpy(y, kd_y.data(), kd_y.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
  SculptorCudaCheckError(cudaMemcpy(z, kd_z.data(), kd_z.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

  return ret;
}

void KdTreeCPU::FindNearestRecursive(std::vector<glm::vec3>::iterator begin,
                                     std::vector<glm::vec3>::iterator end,
                                     int level) {
  if (end <= begin)
    return;
  auto mid = begin + (end - begin) / 2;
  if (auto dist_to_mid = DistFromQuery(*mid); dist_to_mid < best_distance_) {
    best_distance_ = dist_to_mid;
    closest_node_ = mid;
  }
  if (begin + 1 == end)
    return;
  auto diff = reinterpret_cast<float*>(&query_point_)[level] -
              reinterpret_cast<float*>(&(*mid))[level];
  level = level == 2 ? 0 : (level + 1);
  if (diff > -kEps) {
    FindNearestRecursive(mid + 1, end, level);
    if (diff < best_distance_) {
      FindNearestRecursive(begin, mid, level);
      return;
    }
  }
  if (diff < kEps) {
    FindNearestRecursive(begin, mid, level);
    if (diff <= -kEps && -diff < best_distance_)
      FindNearestRecursive(mid + 1, end, level);
  }
}
}  // namespace Sculptor
