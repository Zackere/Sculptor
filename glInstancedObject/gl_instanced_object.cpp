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
  glGenBuffers(1, &positions_gl_buffer_);
  glVertexAttribDivisor(
      glGetAttribLocation(reference_model_->GetShader(), "offset"), 1);

  positions_ = shape_generator_->Generate(nobjects);
  glBindBuffer(GL_ARRAY_BUFFER, positions_gl_buffer_);
  glBufferData(GL_ARRAY_BUFFER, positions_.size() * 3 * sizeof(float),
               positions_.data(), GL_STATIC_DRAW);

  auto materialOffsetsID =
      glGetAttribLocation(reference_model_->GetShader(), "offset");
  glEnableVertexAttribArray(materialOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, positions_gl_buffer_);
  glVertexAttribPointer(materialOffsetsID, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void glInstancedObject::Render(const glm::mat4& vp) const {
  reference_model_->Enable();
  glUniformMatrix4fv(glGetUniformLocation(reference_model_->GetShader(), "mvp"),
                     1, GL_FALSE, &vp[0][0]);
  glBindTexture(GL_TEXTURE_2D, reference_model_->GetTexture());
  glDrawArraysInstanced(GL_TRIANGLES, 0,
                        reference_model_->GetNumberOfModelVertices(),
                        positions_.size());
}

}  // namespace Sculptor
