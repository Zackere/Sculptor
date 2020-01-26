#pragma once

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <memory>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class MatrixApplierBase;
class ShapeGeneratorBase;
class glObject;

class glInstancedObject {
 public:
  glInstancedObject(int ninstances_init,
                    int ninstances_max,
                    std::unique_ptr<glObject> reference_model,
                    std::unique_ptr<ShapeGeneratorBase> shape_generator,
                    std::unique_ptr<MatrixApplierBase> matrix_applier);
  ~glInstancedObject();

  void Render(glm::mat4 const& vp) const;
  void Transform(glm::mat4 const& m);
  int GetNumberOfInstances() const { return x_positions_buffer_.GetSize(); }

 private:
  std::unique_ptr<glObject> reference_model_;
  std::unique_ptr<ShapeGeneratorBase> shape_generator_;
  CudaGraphicsResource<float> x_positions_buffer_;
  CudaGraphicsResource<float> y_positions_buffer_;
  CudaGraphicsResource<float> z_positions_buffer_;
  std::unique_ptr<MatrixApplierBase> matrix_applier_;
};
}  // namespace Sculptor
