#include "cuda_graphics_resource.hpp"

namespace Sculptor {
namespace {
cudaGraphicsResource* Get(GLuint gl_buffer) {
  cudaGraphicsResource* ret = nullptr;
  cudaGraphicsGLRegisterBuffer(&ret, gl_buffer, cudaGraphicsMapFlagsNone);
  return ret;
}
void Destroy(cudaGraphicsResource* res) {
  cudaGraphicsUnregisterResource(res);
}
}  // namespace
CudaGraphicsResource::CudaGraphicsResource()
    : cuda_resource_(nullptr, Destroy) {
  glGenBuffers(1, &gl_buffer_);
  cuda_resource_.reset(Get(gl_buffer_));
}

CudaGraphicsResource::~CudaGraphicsResource() {
  cuda_resource_.reset(nullptr);
  glDeleteBuffers(1, &gl_buffer_);
}
}  // namespace Sculptor
