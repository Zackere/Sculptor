// Copyright 2020 Wojciech Replin. All rights reserved.

#include "gl_instanced_object.hpp"

#include <utility>
#include <vector>

#include "../glObject/gl_object.hpp"
#include "../matrixApplier/matrix_applier_base.hpp"
#include "../modelProvider/model_provider_base.hpp"
#include "../shaderProgram/shader_program_base.hpp"
#include "../shapeGenerator/shape_generator_base.hpp"
#include "../textureProvider/texture_provider_base.hpp"

namespace Sculptor {
glInstancedObject::glInstancedObject(
    int ninstances_init,
    int ninstances_max,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<ShapeGeneratorBase> shape_generator,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : reference_model_(std::move(reference_model)),
      shape_generator_(std::move(shape_generator)),
      x_positions_buffer_(ninstances_max),
      y_positions_buffer_(ninstances_max),
      z_positions_buffer_(ninstances_max),
      matrix_applier_(std::move(matrix_applier)) {
  std::vector<float> x_positions, y_positions, z_positions;
  x_positions.reserve(shape_generator_->GetNumberOfOutputs(ninstances_init));
  y_positions.reserve(x_positions.capacity());
  z_positions.reserve(x_positions.capacity());
  for (auto& v : shape_generator_->Generate(ninstances_init)) {
    x_positions.emplace_back(v.x);
    y_positions.emplace_back(v.y);
    z_positions.emplace_back(v.z);
  }

  x_positions_buffer_.SetData(x_positions.data(), x_positions.size());
  y_positions_buffer_.SetData(y_positions.data(), y_positions.size());
  z_positions_buffer_.SetData(z_positions.data(), z_positions.size());

  auto materialXOffsetsID =
      glGetAttribLocation(reference_model_->GetShader(), "offset_x");
  glVertexAttribDivisor(materialXOffsetsID, 1);
  glEnableVertexAttribArray(materialXOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, x_positions_buffer_.GetGLBuffer());
  glVertexAttribPointer(materialXOffsetsID, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

  auto materialYOffsetsID =
      glGetAttribLocation(reference_model_->GetShader(), "offset_y");
  glVertexAttribDivisor(materialYOffsetsID, 1);
  glEnableVertexAttribArray(materialYOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, y_positions_buffer_.GetGLBuffer());
  glVertexAttribPointer(materialYOffsetsID, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

  auto materialZOffsetsID =
      glGetAttribLocation(reference_model_->GetShader(), "offset_z");
  glVertexAttribDivisor(materialZOffsetsID, 1);
  glEnableVertexAttribArray(materialZOffsetsID);
  glBindBuffer(GL_ARRAY_BUFFER, z_positions_buffer_.GetGLBuffer());
  glVertexAttribPointer(materialZOffsetsID, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
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
  matrix_applier_->Apply(x_positions_buffer_.GetCudaResource(),
                         y_positions_buffer_.GetCudaResource(),
                         z_positions_buffer_.GetCudaResource(),
                         x_positions_buffer_.GetSize(), m);
}

void glInstancedObject::AddInstances(std::vector<glm::vec3> const& instances) {
  std::vector<float> coords(instances.size());
  for (auto i = 0u; i < instances.size(); ++i)
    coords[i] = instances[i].x;
  x_positions_buffer_.Append(coords.data(), coords.size());
  for (auto i = 0u; i < instances.size(); ++i)
    coords[i] = instances[i].y;
  y_positions_buffer_.Append(coords.data(), coords.size());
  for (auto i = 0u; i < instances.size(); ++i)
    coords[i] = instances[i].z;
  z_positions_buffer_.Append(coords.data(), coords.size());
}

void glInstancedObject::SetShader(std::unique_ptr<ShaderProgramBase> shader) {
  reference_model_->SetShader(std::move(shader));
}

}  // namespace Sculptor
