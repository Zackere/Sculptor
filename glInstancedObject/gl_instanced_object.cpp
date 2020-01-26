#include "gl_instanced_object.hpp"

#include <utility>
#include <vector>

#include "../glObject/gl_object.hpp"
#include "../matrixApplier/matrix_applier_base.hpp"
#include "../modelProvider/model_provider_base.hpp"
#include "../shaderProvider/shader_provider_base.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"
#include "../textureProvider/texture_provider_base.hpp"

namespace Sculptor {
glInstancedObject::glInstancedObject(
    int nobjects_start,
    int nobjects_max,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<ShapeGeneratorBase> shape_generator,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : reference_model_(std::move(reference_model)),
      shape_generator_(std::move(shape_generator)),
      positions_buffer_(nullptr),
      matrix_applier_(std::move(matrix_applier)) {
  glVertexAttribDivisor(
      glGetAttribLocation(reference_model_->GetShader(), "offset"), 1);

  std::vector<glm::vec3> positions = shape_generator_->Generate(nobjects_start);
  positions_buffer_ =
      std::make_unique<CudaGraphicsResource<glm::vec3>>(nobjects_max);
  positions_buffer_->SetData(positions.data(), positions.size());

  auto materialOffsetsID =
      glGetAttribLocation(reference_model_->GetShader(), "offset");
  glEnableVertexAttribArray(materialOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, positions_buffer_->GetGLBuffer());
  glVertexAttribPointer(materialOffsetsID, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

glInstancedObject::~glInstancedObject() = default;

void glInstancedObject::Render(glm::mat4 const& vp) const {
  reference_model_->Enable();
  glUniformMatrix4fv(glGetUniformLocation(reference_model_->GetShader(), "mvp"),
                     1, GL_FALSE, &vp[0][0]);
  glBindTexture(GL_TEXTURE_2D, reference_model_->GetTexture());
  glDrawArraysInstanced(GL_TRIANGLES, 0,
                        reference_model_->GetNumberOfModelVertices(),
                        GetNumberOfInstances());
}

void glInstancedObject::Transform(glm::mat4 const& m) {
  reference_model_->Transform(m);
  matrix_applier_->Apply(positions_buffer_->GetCudaResource(),
                         positions_buffer_->GetSize(), m);
}

}  // namespace Sculptor
