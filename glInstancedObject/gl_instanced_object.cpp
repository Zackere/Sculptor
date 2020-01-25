#include "gl_instanced_object.hpp"

#include <utility>

#include "../matrixApplier/matrix_applier_base.hpp"
#include "../modelProvider/model_provider_base.hpp"
#include "../shaderProvider/shader_provider_base.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"
#include "../textureProvider/texture_provider_base.hpp"

namespace Sculptor {
glInstancedObject::glInstancedObject(
    int nobjects,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<ShapeGeneratorBase> shape_generator,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : reference_model_(std::move(reference_model)),
      shape_generator_(std::move(shape_generator)),
      matrix_applier_(std::move(matrix_applier)) {
  glVertexAttribDivisor(
      glGetAttribLocation(reference_model_->GetShader(), "offset"), 1);

  auto positions = shape_generator_->Generate(nobjects);
  positions_buffer_ = std::make_unique<CudaGraphicsResource>(
      positions.data(), positions.size() * 3 * sizeof(float));

  auto materialOffsetsID =
      glGetAttribLocation(reference_model_->GetShader(), "offset");
  glEnableVertexAttribArray(materialOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, positions_buffer_->GetGLBuffer());
  glVertexAttribPointer(materialOffsetsID, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void glInstancedObject::Render(glm::mat4 const& vp) const {
  reference_model_->Enable();
  glUniformMatrix4fv(glGetUniformLocation(reference_model_->GetShader(), "mvp"),
                     1, GL_FALSE, &vp[0][0]);
  glBindTexture(GL_TEXTURE_2D, reference_model_->GetTexture());
  glDrawArraysInstanced(GL_TRIANGLES, 0,
                        reference_model_->GetNumberOfModelVertices(),
                        positions_buffer_->GetSize() / sizeof(glm::vec3));
}

void glInstancedObject::Transform(glm::mat4 const& m) {
  reference_model_->Transform(m);
  matrix_applier_->Apply(positions_buffer_->GetCudaResource(), m);
}

}  // namespace Sculptor
