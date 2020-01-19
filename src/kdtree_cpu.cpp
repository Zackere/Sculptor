#include "../include/kdtree_cpu.hpp"

#include <algorithm>
#include <execution>
#include <iostream>
#include <thread>

namespace Sculptor {
namespace {
template <typename RandomIt>
void ContructKd(RandomIt begin, RandomIt end, int level) {
  if (end <= begin)
    return;
  auto mid = begin + (end - begin) / 2;
  switch (level) {
    default:
      throw 0;
    case 0:
      std::nth_element(begin, mid, end, [](auto const& v1, auto const& v2) {
        return v1.x < v2.x;
      });
      level = 1;
      break;
    case 1:
      std::nth_element(begin, mid, end, [](auto const& v1, auto const& v2) {
        return v1.y < v2.y;
      });
      level = 2;
      break;
    case 2:
      std::nth_element(begin, mid, end, [](auto const& v1, auto const& v2) {
        return v1.z < v2.z;
      });
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

int KdTreeCPU::FindNearest(std::vector<glm::vec3> const& kd_tree,
                           glm::vec3 const& query_point) {
  query_point_ = query_point;
  kd_tree_ = &kd_tree;
  cur_best_index_ = kd_tree.size() / 2;
  cur_best_distance_ = distance(query_point, kd_tree[cur_best_index_]);
  FindNearestInKd(0, kd_tree.size(), 0);
  kd_tree_ = nullptr;
  return cur_best_index_;
}

int KdTreeCPU::Find(std::vector<glm::vec3> const& kd_tree,
                    glm::vec3 const& query_point) {
  // TODO(zackere) : fix FindInKd method and use it instead of this
  auto ret = FindNearest(kd_tree, query_point);
  if (cur_best_distance_ < 0.001)
    return ret;
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
    case 0:
      if (query_point_.x > (*kd_tree_)[mid].x) {
        FindNearestInKd(mid + 1, end, 1);
        if (query_point_.x - (*kd_tree_)[mid].x < cur_best_distance_)
          FindNearestInKd(begin, mid, 1);
      } else {
        FindNearestInKd(begin, mid, 1);
        if ((*kd_tree_)[mid].x - query_point_.x < cur_best_distance_)
          FindNearestInKd(mid + 1, end, 1);
      }
      break;
    case 1:
      if (query_point_.y > (*kd_tree_)[mid].y) {
        FindNearestInKd(mid + 1, end, 2);
        if (query_point_.y - (*kd_tree_)[mid].y < cur_best_distance_)
          FindNearestInKd(begin, mid, 2);
      } else {
        FindNearestInKd(begin, mid, 2);
        if ((*kd_tree_)[mid].y - query_point_.y < cur_best_distance_)
          FindNearestInKd(mid + 1, end, 2);
      }
      break;
    case 2:
      if (query_point_.z > (*kd_tree_)[mid].z) {
        FindNearestInKd(mid + 1, end, 0);
        if (query_point_.z - (*kd_tree_)[mid].z < cur_best_distance_)
          FindNearestInKd(begin, mid, 0);
      } else {
        FindNearestInKd(begin, mid, 0);
        if ((*kd_tree_)[mid].z - query_point_.z < cur_best_distance_)
          FindNearestInKd(mid + 1, end, 0);
      }
      break;
  }
}

int KdTreeCPU::FindInKd(int begin, int end, int level) const {
  if (end <= begin)
    return -1;
  auto mid = (begin + end) / 2;
  float dist_to_mid = glm::distance(query_point_, (*kd_tree_)[mid]);
  if (dist_to_mid < 0.00001)
    return mid;
  if (begin + 1 == end)
    return -1;
  switch (level) {
    default:
      throw 0;
    case 0:
      if (query_point_.x > (*kd_tree_)[mid].x)
        return FindInKd(mid + 1, end, 1);
      else
        return FindInKd(begin, mid, 1);
    case 1:
      if (query_point_.y > (*kd_tree_)[mid].y)
        return FindInKd(mid + 1, end, 2);
      else
        return FindInKd(begin, mid, 2);
    case 2:
      if (query_point_.z > (*kd_tree_)[mid].z)
        return FindInKd(mid + 1, end, 0);
      else
        return FindInKd(begin, mid, 0);
  }
}
}  // namespace Sculptor
