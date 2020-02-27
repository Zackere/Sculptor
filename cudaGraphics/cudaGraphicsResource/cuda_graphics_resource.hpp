// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <memory>
#include <vector>

#include "../../util/cudaCheckError.hpp"

namespace Sculptor {
template <typename ContentType>
class CudaGraphicsResource {
 public:
  explicit CudaGraphicsResource(unsigned capacity);
  ~CudaGraphicsResource();

  auto GetGLBuffer() { return gl_buffer_; }
  auto* GetCudaResource() { return cuda_resource_.get(); }
  std::vector<ContentType> ToStdVector() const;
  auto GetSize() const { return size_; }

  void SetData(ContentType const* data, unsigned nelems);
  void PopBack(int n = 1) { size_ -= n; }
  unsigned PushBack(ContentType const& elem);
  void Append(ContentType const* data, unsigned nelems);
  unsigned Set(ContentType const& elem, unsigned index);
  ContentType Get(unsigned index);

 private:
  std::unique_ptr<cudaGraphicsResource, cudaError_t (*)(cudaGraphicsResource*)>
      cuda_resource_;
  GLuint gl_buffer_ = 0;
  unsigned size_ = 0;
};

template <typename ContentType>
inline CudaGraphicsResource<ContentType>::CudaGraphicsResource(
    unsigned capacity)
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
std::vector<ContentType> CudaGraphicsResource<ContentType>::ToStdVector()
    const {
  auto* resource = cuda_resource_.get();
  SculptorCudaCheckError(cudaGraphicsMapResources(1, &resource));
  ContentType* v = nullptr;
  size_t num_bytes = 0;
  SculptorCudaCheckError(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&v), &num_bytes, resource));
  std::vector<ContentType> ret(size_);
  cudaMemcpy(ret.data(), v, size_ * sizeof(ContentType),
             cudaMemcpyDeviceToHost);
  SculptorCudaCheckError(cudaGraphicsUnmapResources(1, &resource));
  return ret;
}

template <typename ContentType>
inline void CudaGraphicsResource<ContentType>::SetData(ContentType const* data,
                                                       unsigned nelems) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, nelems * sizeof(ContentType), data);
  size_ = nelems;
}

template <typename ContentType>
unsigned CudaGraphicsResource<ContentType>::PushBack(ContentType const& elem) {
  Append(&elem, 1);
  return size_ - 1;
}

template <typename ContentType>
inline void CudaGraphicsResource<ContentType>::Append(ContentType const* data,
                                                      unsigned nelems) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferSubData(GL_ARRAY_BUFFER, size_ * sizeof(ContentType),
                  nelems * sizeof(ContentType), data);
  size_ += nelems;
}

template <typename ContentType>
inline unsigned CudaGraphicsResource<ContentType>::Set(ContentType const& elem,
                                                       unsigned index) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  glBufferSubData(GL_ARRAY_BUFFER, index * sizeof(ContentType),
                  sizeof(ContentType), &elem);
  return index;
}

template <typename ContentType>
inline ContentType CudaGraphicsResource<ContentType>::Get(unsigned index) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_);
  ContentType* p = reinterpret_cast<ContentType*>(
      glMapBufferRange(GL_ARRAY_BUFFER, index * sizeof(ContentType),
                       sizeof(ContentType), GL_MAP_READ_BIT));
  auto ret = *p;
  glUnmapBuffer(GL_ARRAY_BUFFER);
  return ret;
}
}  // namespace Sculptor
