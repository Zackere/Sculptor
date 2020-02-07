#include "kdtree_gpu_thrust_constructor.hpp"
// clang-format on

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

namespace Sculptor {
namespace {
struct ScaleFunctor {
  __host__ __device__ int operator()(float x) { return scaling_factor * x; }
  __host__ __device__ float operator()(int x) { return descaling_factor * x; }

 private:
  static constexpr int scaling_factor = 2048;
  static constexpr float descaling_factor = 1.f / scaling_factor;
};

template <typename T>
void ConstructRecursive(thrust::device_vector<T>& x,
                        thrust::device_vector<T>& y,
                        thrust::device_vector<T>& z,
                        int begin,
                        int end) {
  if (end <= begin)
    return;
  auto mid = begin + (end - begin) / 2;
  auto zip =
      thrust::make_zip_iterator(thrust::make_tuple(y.begin(), z.begin()));
  thrust::sort_by_key(thrust::device, x.begin() + begin, x.begin() + end,
                      zip + begin);
  ConstructRecursive(y, z, x, begin, mid);
  ConstructRecursive(y, z, x, mid + 1, end);
}
}  // namespace

void KdTreeGPUThrustConstructor::Construct(float* x,
                                           float* y,
                                           float* z,
                                           int size) {
  thrust::device_vector<int> x_int(size), y_int(size), z_int(size);

  thrust::transform(x, x + size, x_int.begin(), ScaleFunctor());
  thrust::transform(y, y + size, y_int.begin(), ScaleFunctor());
  thrust::transform(z, z + size, z_int.begin(), ScaleFunctor());

  ConstructRecursive(x_int, y_int, z_int, 0, size);

  thrust::transform(x_int.begin(), x_int.end(), x, ScaleFunctor());
  thrust::transform(y_int.begin(), y_int.end(), y, ScaleFunctor());
  thrust::transform(z_int.begin(), z_int.end(), z, ScaleFunctor());
}
}  // namespace Sculptor
