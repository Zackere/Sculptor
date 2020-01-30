#include "cudaCheckError.hpp"

namespace Sculptor {
void CudaCheckError(cudaError_t error, char const* file, int line) {
  if (error)
    std::cerr << cudaGetErrorString(error) << '(' << file << ',' << line << ')'
              << std::endl;
}
}  // namespace Sculptor
