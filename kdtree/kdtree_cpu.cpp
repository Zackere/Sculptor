#include "kdtree_cpu.hpp"

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <execution>
#include <functional>
#include <set>

namespace Sculptor {
namespace {
constexpr float kEps = 0.001f;

template <typename RandomIt>
void ConstructRecursive(RandomIt begin, RandomIt end, int level) {
  if (end <= begin)
    return;
  switch (level) {
    default:
      throw 0;
    case 0:
      std::sort(std::execution::par, begin, end,
                [](auto const& v1, auto const& v2) { return v1.x < v2.x; });
      level = 1;
      break;
    case 1:
      std::sort(std::execution::par, begin, end,
                [](auto const& v1, auto const& v2) { return v1.y < v2.y; });
      level = 2;
      break;
    case 2:
      std::sort(std::execution::par, begin, end,
                [](auto const& v1, auto const& v2) { return v1.z < v2.z; });
      level = 0;
      break;
  }
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

void KdTreeCPU::RemoveNearest(CudaGraphicsResource<float>& x,
                              CudaGraphicsResource<float>& y,
                              CudaGraphicsResource<float>& z,
                              CudaGraphicsResource<glm::vec3>& query_points,
                              float threshold) {
  if (!x.GetSize())
    return;

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

  std::vector<Zip> kd(x.GetSize());
  for (int i = 0; i < x.GetSize(); ++i)
    kd[i] = {kd_x.data() + i, kd_y.data() + i, kd_z.data() + i};
  std::set<std::vector<Zip>::iterator, std::greater<>> to_be_removed;
  auto const threshold_squared = threshold * threshold;
  for (auto const& v : queries) {
    closest_node_ = kd.begin() + (kd.end() - kd.begin()) / 2;
    query_point_ = v;
    best_distance_squared_ = DistFromQuery(*closest_node_);
    FindNearestRecursive(kd.begin(), kd.end(), 0);
    if (best_distance_squared_ < threshold_squared)
      to_be_removed.insert(closest_node_);
  }
  for (auto it : to_be_removed) {
    *it->x = *kd.back().x;
    *it->y = *kd.back().y;
    *it->z = *kd.back().z;
    kd.pop_back();
  }

  x.SetData(kd_x.data(), kd.size());
  y.SetData(kd_y.data(), kd.size());
  z.SetData(kd_z.data(), kd.size());

  cudaGraphicsUnmapResources(1, &query_res);
  cudaGraphicsUnmapResources(1, &z_res);
  cudaGraphicsUnmapResources(1, &y_res);
  cudaGraphicsUnmapResources(1, &x_res);
}

void KdTreeCPU::FindNearestRecursive(std::vector<Zip>::iterator begin,
                                     std::vector<Zip>::iterator end,
                                     int level) {
  if (end <= begin)
    return;
  auto mid = begin + (end - begin) / 2;
  float dist_to_mid = DistFromQuery(*mid);
  if (dist_to_mid < best_distance_squared_) {
    best_distance_squared_ = dist_to_mid;
    closest_node_ = mid;
  }
  if (begin + 1 == end)
    return;
  switch (level) {
    default:
      throw 0;
    case 0: {
      float diff = query_point_.x - *mid->x;
      if (diff > -kEps) {
        FindNearestRecursive(mid + 1, end, 1);
        if (diff < best_distance_squared_) {
          FindNearestRecursive(begin, mid, 1);
          break;
        }
      }
      if (diff < kEps) {
        FindNearestRecursive(begin, mid, 1);
        if (diff <= -kEps && -diff < best_distance_squared_) {
          FindNearestRecursive(mid + 1, end, 1);
          break;
        }
      }
    } break;
    case 1: {
      float diff = query_point_.y - *mid->y;
      if (diff > -kEps) {
        FindNearestRecursive(mid + 1, end, 2);
        if (diff < best_distance_squared_) {
          FindNearestRecursive(begin, mid, 2);
          break;
        }
      }
      if (diff < kEps) {
        FindNearestRecursive(begin, mid, 2);
        if (diff <= -kEps && -diff < best_distance_squared_) {
          FindNearestRecursive(mid + 1, end, 2);
          break;
        }
      }
    } break;
    case 2: {
      float diff = query_point_.z - *mid->z;
      if (diff > -kEps) {
        FindNearestRecursive(mid + 1, end, 0);
        if (diff < best_distance_squared_) {
          FindNearestRecursive(begin, mid, 0);
          break;
        }
      }
      if (diff < kEps) {
        FindNearestRecursive(begin, mid, 0);
        if (diff <= -kEps && -diff < best_distance_squared_) {
          FindNearestRecursive(mid + 1, end, 0);
          break;
        }
      }
    } break;
  }
}
}  // namespace Sculptor
