#include "../include/sculpting_material.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string_view>

#include "../include/objloader.hpp"

namespace Sculptor {
SculptingMaterial::SculptingMaterial(MaterialType material_type,
                                     InitialShape initial_shape,
                                     int size)
    : reference_model_(), reference_model_gl_() {
  glGenBuffers(1, &reference_model_gl_.verticies);
  glGenBuffers(1, &reference_model_gl_.uvs);
  glGenBuffers(1, &reference_model_gl_.normals);
  glGenBuffers(1, &offsets_buffer);

  switch (material_type) {
    case MaterialType::CUBE:
      OBJLoader::LoadOBJ("../Sculptor/models/Cube.obj",
                         reference_model_.verticies, reference_model_.uvs,
                         reference_model_.normals);
      break;
    case MaterialType::SPHERE:
      OBJLoader::LoadOBJ("../Sculptor/models/Sphere.obj",
                         reference_model_.verticies, reference_model_.uvs,
                         reference_model_.normals);
      break;
  }
  const auto scale = 1.f / size;
  for (auto& v : reference_model_.verticies)
    v *= scale;
  Reset(initial_shape, size);
}

void SculptingMaterial::Reset(InitialShape new_shape, int size) {
  const auto step = 2.f / size;
  const auto start = step / 2 - 1;
  switch (new_shape) {
    case InitialShape::CUBE:
      for (auto x = start; x < 1.f; x += step)
        for (auto y = start; y < 1.f; y += step)
          for (auto z = start; z < 1.f; z += step)
            offsets.emplace_back(x, y, z);

      glBindBuffer(GL_ARRAY_BUFFER, offsets_buffer);
      glBufferData(GL_ARRAY_BUFFER, offsets.size() * 3 * sizeof(float),
                   offsets.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.verticies);
      glBufferData(GL_ARRAY_BUFFER,
                   reference_model_.verticies.size() * 3 * sizeof(float),
                   reference_model_.verticies.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.uvs);
      glBufferData(GL_ARRAY_BUFFER,
                   reference_model_.uvs.size() * 2 * sizeof(float),
                   reference_model_.uvs.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, reference_model_gl_.normals);
      glBufferData(GL_ARRAY_BUFFER,
                   reference_model_.normals.size() * 3 * sizeof(float),
                   reference_model_.normals.data(), GL_STATIC_DRAW);
      break;
  }
}

void SculptingMaterial::RemoveAt(unsigned index) {
  if (index >= offsets.size()) {
    std::cerr << "Index is out of bounds : index: " << index << std::endl;
    return;
  }
  if (offsets.empty())
    return;
  if (index + 1 < offsets.size()) {
    offsets[index] = offsets.back();
    glBindBuffer(GL_ARRAY_BUFFER, offsets_buffer);
    auto* p = reinterpret_cast<glm::vec3*>(
        glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
    p[index] = p[offsets.size() - 1];
    glUnmapBuffer(GL_ARRAY_BUFFER);
  }
  offsets.pop_back();
}
}  // namespace Sculptor
