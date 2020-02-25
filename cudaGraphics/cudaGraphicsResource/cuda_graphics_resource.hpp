// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <memory>

#include "../../util/cudaCheckError.hpp"

namespace Sculptor {
template <typename ContentType>
class CudaGraphicsResource {
 public:
  explicit CudaGraphicsResource(size_t capacity);
  ~CudaGraphicsResource();

  auto GetGLBuffer() { return gl_buffer_; }
  auto* GetCudaResource() { return cuda_resource_.get(); }
  auto GetSize() const { return size_; }

  void SetData(ContentType const* data, size_t nelems);
  void PopBack(int n = 1) { size_ -= n; }
  void PushBack(ContentType const& elem);
  void Append(ContentType const* data, size_t nelems);
  void Set(ContentType const& elem, size_t index);

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
  SculptorCudaCheckError(
      cudaGraphicsGLRegisterBuffer(&res, gl_buffer_, cudaGraphicsMapFlagsNone));
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
  Append(&elem, 1);
}

template <typename ContentType>
inline void CudaGraphicsResource<ContentType>::Append(ContentType const* data,
                                                      size_t nelems) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferSubData(GL_ARRAY_BUFFER, size_ * sizeof(ContentType),
                  nelems * sizeof(ContentType), data);
  size_ += nelems;
}

template <typename ContentType>
inline void CudaGraphicsResource<ContentType>::Set(ContentType const& elem,
                                                   size_t index) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferSubData(GL_ARRAY_BUFFER, index * sizeof(ContentType),
                  sizeof(ContentType), &elem);
}
}  // namespace Sculptor
