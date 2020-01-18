#include "../include/kdtree_cpu.hpp"

#include <algorithm>
#include <execution>
#include <thread>

namespace Sculptor {
namespace {
template <typename RandomIt>
void construct_kd(RandomIt begin, RandomIt end, int level) {
  if (end - begin <= 1)
    return;
  switch (level % 3) {
    case 0:
      std::sort(std::execution::par, begin, end,
                [](auto const& v1, auto const& v2) { return v1.x < v2.x; });
      break;
    case 1:
      std::sort(std::execution::par, begin, end,
                [](auto const& v1, auto const& v2) { return v1.y < v2.y; });
      break;
    case 2:
      std::sort(std::execution::par, begin, end,
                [](auto const& v1, auto const& v2) { return v1.z < v2.z; });
      break;
  }
  RandomIt mid = begin + (end - begin) / 2;
  ++level;
  construct_kd(begin, mid, level);
  construct_kd(mid + 1, end, level);
}
}  // namespace
void KdTreeCPU::Construct(std::vector<glm::vec3>& v) {
  construct_kd(v.begin(), v.end(), 0);
}
}  // namespace Sculptor
