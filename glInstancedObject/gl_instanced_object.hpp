#pragma once

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <memory>
#include <string_view>
#include <vector>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "../glObject/gl_object.hpp"

namespace Sculptor {
class ModelProviderBase;
class ShaderProviderBase;
class MatrixApplierBase;
class TextureProviderBase;
class ShapeGeneratorBase;

class glInstancedObject {
 public:
  glInstancedObject(int nobjects,
                    std::unique_ptr<glObject> reference_model,
                    std::unique_ptr<ShapeGeneratorBase> shape_generator,
                    std::unique_ptr<MatrixApplierBase> matrix_applier);

  void Render(glm::mat4 const& vp) const;
  void Transform(glm::mat4 const& m);

 private:
  std::unique_ptr<glObject> reference_model_;
  std::unique_ptr<ShapeGeneratorBase> shape_generator_ = nullptr;
  std::unique_ptr<CudaGraphicsResource> positions_buffer_ = nullptr;
  std::unique_ptr<MatrixApplierBase> matrix_applier_ = nullptr;
};
}  // namespace Sculptor
