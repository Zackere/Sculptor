#pragma once

#include "../include/kdtree.hpp"

namespace Sculptor {
class KdTreeCPU : public KdTree {
 public:
  ~KdTreeCPU() override = default;
  void Construct(std::vector<glm::vec3>& v) override;
  int FindNearest(std::vector<glm::vec3> const& kd_tree,
                  glm::vec3 const& query_point) override;
  int Find(std::vector<glm::vec3> const& kd_tree,
           glm::vec3 const& query_point) override;
  float GetDistanceToLastFound() const override { return cur_best_distance_; }

 private:
  void FindNearestInKd(int begin, int end, int level);
  int FindInKd(int begin, int end, int level);

  glm::vec3 query_point_ = glm::vec3(0, 0, 0);
  glm::vec3 cur_best_ = glm::vec3(0, 0, 0);
  unsigned cur_best_index_ = 0;
  float cur_best_distance_ = 0;
  std::vector<glm::vec3> const* kd_tree_ = nullptr;
};
}  // namespace Sculptor
