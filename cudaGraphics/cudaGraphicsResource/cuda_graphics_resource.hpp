#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <memory>

namespace Sculptor {
template <typename ContentType>
class CudaGraphicsResource {
 public:
  CudaGraphicsResource(size_t capacity);
  ~CudaGraphicsResource();

  auto GetGLBuffer() { return gl_buffer_; }
  auto* GetCudaResource() { return cuda_resource_.get(); }
  auto GetSize() { return size_; }

  void SetData(ContentType const* data, size_t nelems);
  void PopBack() { --size_; };
  void PushBack(ContentType const& elem);

 private:
  std::unique_ptr<cudaGraphicsResource, cudaError_t (*)(cudaGraphicsResource*)>
      cuda_resource_;
  GLuint gl_buffer_ = 0;
  int size_ = 0;
};

template <typename ContentType>
inline CudaGraphicsResource<ContentType>::CudaGraphicsResource(size_t capacity)
    : cuda_resource_(nullptr, cudaGraphicsUnregisterResource) {
  glGenBuffers(1, &gl_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferData(GL_ARRAY_BUFFER, capacity * sizeof(ContentType), nullptr,
               GL_STATIC_DRAW);
  cudaGraphicsResource* res = nullptr;
  cudaGraphicsGLRegisterBuffer(&res, gl_buffer_, cudaGraphicsMapFlagsNone);
  cuda_resource_.reset(res);
}

template <typename ContentType>
inline CudaGraphicsResource<ContentType>::~CudaGraphicsResource() {
  glDeleteBuffers(1, &gl_buffer_);
}

template <typename ContentType>
inline void CudaGraphicsResource<ContentType>::SetData(ContentType const* data,
                                                       size_t nelems) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, nelems * sizeof(ContentType), data);
  size_ = nelems;
}

template <typename ContentType>
inline void CudaGraphicsResource<ContentType>::PushBack(
    ContentType const& elem) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferSubData(GL_ARRAY_BUFFER, size_ * sizeof(ContentType),
                  sizeof(ContentType), &elem);
  ++size_;
}
}  // namespace Sculptor
