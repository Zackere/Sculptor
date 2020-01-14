#include "../include/sculpting_material.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string_view>

namespace Sculptor {
namespace {
bool LoadOBJ(std::string_view path,
             std::vector<glm::vec3>& out_vertices,
             std::vector<glm::vec2>& out_uvs,
             std::vector<glm::vec3>& out_normals) {
  std::cout << "Loading OBJ file " << path << '\n';

  std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
  std::vector<glm::vec3> temp_vertices;
  std::vector<glm::vec2> temp_uvs;
  std::vector<glm::vec3> temp_normals;

  FILE* file = std::fopen(path.data(), "r");
  if (file == NULL) {
    printf(
        "Impossible to open the file ! Are you in the right path ? See "
        "Tutorial 1 for details\n");
    return false;
  }

  while (true) {
    char lineHeader[128];
    // read the first word of the line
    int res = fscanf(file, "%s", lineHeader);
    if (res == EOF)
      break;  // EOF = End Of File. Quit the loop.

    // else : parse lineHeader

    if (strcmp(lineHeader, "v") == 0) {
      glm::vec3 vertex;
      if (fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z) != 3)
        std::cerr << "fscanf error at" << __LINE__ << __FILE__ << std::endl;
      temp_vertices.push_back(vertex);
    } else if (strcmp(lineHeader, "vt") == 0) {
      glm::vec2 uv;
      if (fscanf(file, "%f %f\n", &uv.x, &uv.y) != 2)
        std::cerr << "fscanf error at" << __LINE__ << __FILE__ << std::endl;
      uv.y = -uv.y;  // Invert V coordinate since we will only use DDS texture,
                     // which are inverted. Remove if you want to use TGA or BMP
                     // loaders.
      temp_uvs.push_back(uv);
    } else if (strcmp(lineHeader, "vn") == 0) {
      glm::vec3 normal;
      if (fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z) != 3)
        std::cerr << "fscanf error at" << __LINE__ << __FILE__ << std::endl;
      temp_normals.push_back(normal);
    } else if (strcmp(lineHeader, "f") == 0) {
      std::string vertex1, vertex2, vertex3;
      unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
      int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n",
                           &vertexIndex[0], &uvIndex[0], &normalIndex[0],
                           &vertexIndex[1], &uvIndex[1], &normalIndex[1],
                           &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
      if (matches != 9) {
        printf(
            "File can't be read by our simple parser :-( Try exporting with "
            "other options\n");
        fclose(file);
        return false;
      }
      vertexIndices.push_back(vertexIndex[0]);
      vertexIndices.push_back(vertexIndex[1]);
      vertexIndices.push_back(vertexIndex[2]);
      uvIndices.push_back(uvIndex[0]);
      uvIndices.push_back(uvIndex[1]);
      uvIndices.push_back(uvIndex[2]);
      normalIndices.push_back(normalIndex[0]);
      normalIndices.push_back(normalIndex[1]);
      normalIndices.push_back(normalIndex[2]);
    } else {
      // Probably a comment, eat up the rest of the line
      char stupidBuffer[1000];
      if (fgets(stupidBuffer, 1000, file) != stupidBuffer)
        std::cerr << "fgets error at" << __LINE__ << __FILE__ << std::endl;
    }
  }

  // For each vertex of each triangle
  for (unsigned int i = 0; i < vertexIndices.size(); i++) {
    // Get the indices of its attributes
    unsigned int vertexIndex = vertexIndices[i];
    unsigned int uvIndex = uvIndices[i];
    unsigned int normalIndex = normalIndices[i];

    // Get the attributes thanks to the index
    glm::vec3 vertex = temp_vertices[vertexIndex - 1];
    glm::vec2 uv = temp_uvs[uvIndex - 1];
    glm::vec3 normal = temp_normals[normalIndex - 1];

    // Put the attributes in buffers
    out_vertices.push_back(vertex);
    out_uvs.push_back(uv);
    out_normals.push_back(normal);
  }
  fclose(file);
  return true;
}
}  // namespace
SculptingMaterial::SculptingMaterial(MaterialType material_type,
                                     InitialShape initial_shape,
                                     int size)
    : reference_model_(), material_() {
  glGenBuffers(1, &material_.verticies);
  glGenBuffers(1, &material_.uvs);
  glGenBuffers(1, &material_.normals);

  switch (material_type) {
    case MaterialType::CUBE:
      LoadOBJ("../Sculptor/models/Cube.obj", reference_model_.verticies,
              reference_model_.uvs, reference_model_.normals);
      break;
    case MaterialType::SPHERE:
      LoadOBJ("../Sculptor/models/Sphere.obj", reference_model_.verticies,
              reference_model_.uvs, reference_model_.normals);
      break;
  }
  Reset(initial_shape, size);
}

void SculptingMaterial::Reset(InitialShape new_shape, int size) {
  std::vector<glm::vec3> verticies = {}, normals = {};
  verticies.reserve(reference_model_.verticies.size() * size * size * size);
  normals.reserve(reference_model_.normals.size() * size * size * size);
  std::vector<glm::vec2> uvs = {};
  uvs.reserve(reference_model_.uvs.size() * size * size * size);
  material_.offsets.clear();
  material_.offsets.reserve(size * size * size);

  auto sizef = static_cast<float>(size - 1);
  auto scale = 1.f / size;
  switch (new_shape) {
    case InitialShape::CUBE:
      for (auto x = -sizef; x <= sizef; x += 2)
        for (auto y = -sizef; y <= sizef; y += 2)
          for (auto z = -sizef; z <= sizef; z += 2) {
            material_.offsets.emplace_back(x, y, z);
            for (auto const& vec : reference_model_.verticies)
              verticies.emplace_back((vec + material_.offsets.back()) * scale);
            std::copy(reference_model_.normals.begin(),
                      reference_model_.normals.end(),
                      std::back_inserter(normals));
            std::copy(reference_model_.uvs.begin(), reference_model_.uvs.end(),
                      std::back_inserter(uvs));
          }
      nverticies_ = verticies.size();
      glBindBuffer(GL_ARRAY_BUFFER, material_.verticies);
      glBufferData(GL_ARRAY_BUFFER, verticies.size() * 3 * sizeof(float),
                   verticies.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, material_.uvs);
      glBufferData(GL_ARRAY_BUFFER, uvs.size() * 2 * sizeof(float), uvs.data(),
                   GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, material_.normals);
      glBufferData(GL_ARRAY_BUFFER, normals.size() * 3 * sizeof(float),
                   normals.data(), GL_STATIC_DRAW);
      break;
  }
}

void SculptingMaterial::RemoveAt(unsigned index) {
  if (index >= material_.offsets.size()) {
    std::cerr << "Index is out of bounds : index: " << index << std::endl;
    return;
  }
  if (material_.offsets.empty())
    return;

  if (index < material_.offsets.size() - 1) {
    glBindBuffer(GL_ARRAY_BUFFER, material_.verticies);
    auto* v = reinterpret_cast<glm::vec3*>(
        glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
    for (auto i = 0u; i < reference_model_.verticies.size(); ++i)
      v[index * reference_model_.verticies.size() + i] =
          v[(material_.offsets.size() - 1) * reference_model_.verticies.size() +
            i];
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, material_.uvs);
    auto* u = reinterpret_cast<glm::vec2*>(
        glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
    for (auto i = 0u; i < reference_model_.uvs.size(); ++i)
      u[index * reference_model_.uvs.size() + i] =
          u[(material_.offsets.size() - 1) * reference_model_.uvs.size() + i];
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, material_.normals);
    auto* n = reinterpret_cast<glm::vec3*>(
        glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
    for (auto i = 0u; i < reference_model_.normals.size(); ++i)
      n[index * reference_model_.normals.size() + i] =
          n[(material_.offsets.size() - 1) * reference_model_.normals.size() +
            i];
    glUnmapBuffer(GL_ARRAY_BUFFER);
    material_.offsets[index] = material_.offsets.back();
  }
  nverticies_ -= reference_model_.verticies.size();
  material_.offsets.pop_back();
}
}  // namespace Sculptor
