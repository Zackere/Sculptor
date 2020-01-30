#include "kdtree_cpu.hpp"

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <execution>
#include <functional>
#include <set>
#include <utility>

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
void KdTreeCPU::Construct(CudaGraphicsResource<float>& x,
                          CudaGraphicsResource<float>& y,
                          CudaGraphicsResource<float>& z) {
  if (!x.GetSize())
    return;

  auto *x_res = x.GetCudaResource(), *y_res = y.GetCudaResource(),
       *z_res = z.GetCudaResource();

  cudaGraphicsMapResources(1, &x_res);
  cudaGraphicsMapResources(1, &y_res);
  cudaGraphicsMapResources(1, &z_res);

  float *dx, *dy, *dz;
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dx),
                                       &num_bytes, x_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dy),
                                       &num_bytes, y_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dz),
                                       &num_bytes, z_res);

  std::vector<float> kd_x(x.GetSize()), kd_y(y.GetSize()), kd_z(z.GetSize());
  cudaMemcpy(kd_x.data(), dx, x.GetSize() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(kd_y.data(), dy, y.GetSize() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(kd_z.data(), dz, z.GetSize() * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::vector<glm::vec3> kd(x.GetSize());
  for (auto i = 0u; i < kd.size(); ++i)
    kd[i] = {kd_x[i], kd_y[i], kd_z[i]};
  ConstructRecursive(kd.begin(), kd.end(), 0);
  for (auto i = 0u; i < kd.size(); ++i) {
    kd_x[i] = kd[i].x;
    kd_y[i] = kd[i].y;
    kd_z[i] = kd[i].z;
  }

  x.SetData(kd_x.data(), kd_x.size());
  y.SetData(kd_y.data(), kd_y.size());
  z.SetData(kd_z.data(), kd_z.size());

  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);
}

std::vector<glm::vec3> KdTreeCPU::RemoveNearest(
    CudaGraphicsResource<float>& x,
    CudaGraphicsResource<float>& y,
    CudaGraphicsResource<float>& z,
    CudaGraphicsResource<glm::vec3>& query_points,
    float threshold,
    bool construct) {
  if (!x.GetSize())
    return {};

  auto x_res = x.GetCudaResource(), y_res = y.GetCudaResource(),
       z_res = z.GetCudaResource(), query_res = query_points.GetCudaResource();

  cudaGraphicsMapResources(1, &x_res);
  cudaGraphicsMapResources(1, &y_res);
  cudaGraphicsMapResources(1, &z_res);
  cudaGraphicsMapResources(1, &query_res);

  float *dx, *dy, *dz;
  glm::vec3* dq;
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dx),
                                       &num_bytes, x_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dy),
                                       &num_bytes, y_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dz),
                                       &num_bytes, z_res);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dq),
                                       &num_bytes, query_res);

  std::vector<float> kd_x(x.GetSize()), kd_y(y.GetSize()), kd_z(z.GetSize());
  cudaMemcpy(kd_x.data(), dx, x.GetSize() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(kd_y.data(), dy, y.GetSize() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(kd_z.data(), dz, z.GetSize() * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::vector<glm::vec3> queries(query_points.GetSize());
  cudaMemcpy(queries.data(), dq, query_points.GetSize() * sizeof(glm::vec3),
             cudaMemcpyDeviceToHost);

  std::vector<glm::vec3> kd(x.GetSize());
  for (auto i = 0u; i < kd.size(); ++i)
    kd[i] = {kd_x[i], kd_y[i], kd_z[i]};
  if (construct)
    ConstructRecursive(kd.begin(), kd.end(), 0);
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

  x.SetData(kd_x.data(), kd.size());
  y.SetData(kd_y.data(), kd.size());
  z.SetData(kd_z.data(), kd.size());

  cudaGraphicsUnmapResources(1, &query_res);
  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);

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
