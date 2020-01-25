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
CudaGraphicsResource::CudaGraphicsResource(void* data, size_t num_bytes)
    : cuda_resource_(nullptr, Destroy) {
  glGenBuffers(1, &gl_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferData(GL_ARRAY_BUFFER, num_bytes, data, GL_STATIC_DRAW);
  cuda_resource_.reset(Get(gl_buffer_));
  size_ = num_bytes;
}

CudaGraphicsResource::~CudaGraphicsResource() {
  cuda_resource_.reset(nullptr);
  glDeleteBuffers(1, &gl_buffer_);
}
}  // namespace Sculptor
