#pragma once

#include <cuda_runtime.h>

#include <iostream>

namespace Sculptor {
void CudaCheckError(cudaError_t error, char const* file, int line);
#ifndef SculptorCudaCheckError
#define SculptorCudaCheckError(x) CudaCheckError(x, __FILE__, __LINE__)
#endif
}  // namespace Sculptor
