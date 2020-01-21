#include "../include/kdtree_cpu.hpp"

#include <algorithm>
#include <execution>
#include <iostream>
#include <thread>

namespace Sculptor {
namespace {
constexpr float kEps = 0.001;

template <typename RandomIt>
void ContructKd(RandomIt begin, RandomIt end, int level) {
  if (end <= begin)
    return;
  auto mid = begin + (end - begin) / 2;
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
  ContructKd(begin, mid, level);
  ContructKd(mid + 1, end, level);
}
}  // namespace
void KdTreeCPU::Construct(std::vector<glm::vec3>& v) {
  ContructKd(v.begin(), v.end(), 0);
}

std::map<int, float, std::greater<>> KdTreeCPU::FindNearest(
    std::vector<glm::vec3> const& kd_tree,
    std::vector<glm::vec3> const& query_points) {
  std::map<int, float, std::greater<>> ret;
  kd_tree_ = &kd_tree;
  for (auto const& p : query_points) {
    query_point_ = p;
    cur_best_index_ = kd_tree.size() / 2;
    cur_best_distance_ = distance(p, kd_tree[cur_best_index_]);
    FindNearestInKd(0, kd_tree.size(), 0);
    if (auto it = ret.find(cur_best_index_); it != ret.end()) {
      if (it->second > cur_best_distance_)
        it->second = cur_best_distance_;
    } else {
      ret.insert({cur_best_index_, cur_best_distance_});
    }
  }
  kd_tree_ = nullptr;

  return ret;
}

int KdTreeCPU::Find(std::vector<glm::vec3> const& kd_tree,
                    glm::vec3 const& query_point) {
  // TODO(zackere) : fix FindInKd method and use it instead of this
  auto ret = FindNearest(kd_tree, {query_point});
  if (ret.begin()->second < kEps)
    return ret.begin()->first;
  return -1;
}

void KdTreeCPU::FindNearestInKd(int begin, int end, int level) {
  if (end <= begin)
    return;
  auto mid = (begin + end) / 2;
  float dist_to_mid = glm::distance(query_point_, (*kd_tree_)[mid]);
  if (dist_to_mid < cur_best_distance_) {
    cur_best_distance_ = dist_to_mid;
    cur_best_index_ = mid;
    cur_best_ = (*kd_tree_)[mid];
  }
  if (begin + 1 == end)
    return;
  switch (level) {
    default:
      throw 0;
    case 0: {
      float diff = query_point_.x - (*kd_tree_)[mid].x;
      if (diff > -kEps) {
        FindNearestInKd(mid + 1, end, 1);
        if (query_point_.x - (*kd_tree_)[mid].x < cur_best_distance_) {
          FindNearestInKd(begin, mid, 1);
          break;
        }
      }
      if (diff < kEps) {
        FindNearestInKd(begin, mid, 1);
        if (diff <= -kEps &&
            (*kd_tree_)[mid].x - query_point_.x < cur_best_distance_) {
          FindNearestInKd(mid + 1, end, 1);
          break;
        }
      }
    } break;
    case 1: {
      float diff = query_point_.y - (*kd_tree_)[mid].y;
      if (diff > -kEps) {
        FindNearestInKd(mid + 1, end, 2);
        if (query_point_.y - (*kd_tree_)[mid].y < cur_best_distance_) {
          FindNearestInKd(begin, mid, 2);
          break;
        }
      }
      if (diff < kEps) {
        FindNearestInKd(begin, mid, 2);
        if (diff <= -kEps &&
            (*kd_tree_)[mid].y - query_point_.y < cur_best_distance_) {
          FindNearestInKd(mid + 1, end, 2);
          break;
        }
      }
    } break;
    case 2: {
      float diff = query_point_.z > (*kd_tree_)[mid].z;
      if (diff > -kEps) {
        FindNearestInKd(mid + 1, end, 0);
        if (query_point_.z - (*kd_tree_)[mid].z < cur_best_distance_) {
          FindNearestInKd(begin, mid, 0);
          break;
        }
      }
      if (diff < kEps) {
        FindNearestInKd(begin, mid, 0);
        if (diff <= -kEps &&
            (*kd_tree_)[mid].z - query_point_.z < cur_best_distance_) {
          FindNearestInKd(mid + 1, end, 0);
          break;
        }
      }
    } break;
  }
}

int KdTreeCPU::FindInKd(int begin, int end, int level) const {
  if (end <= begin)
    return -1;
  auto mid = (begin + end) / 2;
  float dist_to_mid = glm::distance(query_point_, (*kd_tree_)[mid]);
  if (dist_to_mid < kEps)
    return mid;
  if (begin + 1 == end)
    return -1;
  float diff = 0;
  switch (level) {
    default:
      throw 0;
    case 0:
      diff = query_point_.x - (*kd_tree_)[mid].x;
      level = 1;
      break;
    case 1:
      diff = query_point_.y - (*kd_tree_)[mid].y;
      level = 2;
      break;
    case 2:
      diff = query_point_.z - (*kd_tree_)[mid].z;
      level = 0;
      break;
  }
  int ret = -1;
  if (diff > -kEps)
    ret = FindInKd(begin, mid, level);
  if (diff < kEps && ret == -1)
    ret = FindInKd(begin, mid, level);
  return ret;
}
}  // namespace Sculptor
