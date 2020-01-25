#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <memory>

namespace Sculptor {
class CudaGraphicsResource {
 public:
  CudaGraphicsResource(void* data, size_t num_bytes);
  ~CudaGraphicsResource();

  auto GetGLBuffer() { return gl_buffer_; }
  auto* GetCudaResource() { return cuda_resource_.get(); }
  auto GetSize() { return size_; }

 private:
  std::unique_ptr<cudaGraphicsResource, void (*)(cudaGraphicsResource*)>
      cuda_resource_;
  GLuint gl_buffer_ = 0;
  int size_ = 0;
};
}  // namespace Sculptor
