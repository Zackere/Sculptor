#include "obj_loader.hpp"

#include <iostream>
#include <memory>
#include <string>

namespace Sculptor {
bool ObjLoader::Load(std::string_view path,
                     std::vector<glm::vec3>& out_vertices,
                     std::vector<glm::vec2>& out_uvs,
                     std::vector<glm::vec3>& out_normals) {
  std::cout << "Loading OBJ file " << path << '\n';

  std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
  std::vector<glm::vec3> temp_vertices;
  std::vector<glm::vec2> temp_uvs;
  std::vector<glm::vec3> temp_normals;

  std::unique_ptr<FILE, int (*)(FILE*)> file(std::fopen(path.data(), "r"),
                                             fclose);
  if (!file) {
    std::cerr << "Impossible to open the file ! Are you in the right path ?\n";
    return false;
  }

  while (true) {
    char lineHeader[128];
    // read the first word of the line
    int res = fscanf(file.get(), "%s", lineHeader);
    if (res == EOF)
      break;  // EOF = End Of File. Quit the loop.

    // else : parse lineHeader

    if (strcmp(lineHeader, "v") == 0) {
      glm::vec3 vertex;
      if (fscanf(file.get(), "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z) !=
          3)
        std::cerr << "fscanf error at" << __LINE__ << __FILE__ << std::endl;
      temp_vertices.push_back(vertex);
    } else if (strcmp(lineHeader, "vt") == 0) {
      glm::vec2 uv;
      if (fscanf(file.get(), "%f %f\n", &uv.x, &uv.y) != 2)
        std::cerr << "fscanf error at" << __LINE__ << __FILE__ << std::endl;
      uv.y = -uv.y;  // Invert V coordinate since we will only use DDS texture,
                     // which are inverted. Remove if you want to use TGA or BMP
                     // loaders.
      temp_uvs.push_back(uv);
    } else if (strcmp(lineHeader, "vn") == 0) {
      glm::vec3 normal;
      if (fscanf(file.get(), "%f %f %f\n", &normal.x, &normal.y, &normal.z) !=
          3)
        std::cerr << "fscanf error at" << __LINE__ << __FILE__ << std::endl;
      temp_normals.push_back(normal);
    } else if (strcmp(lineHeader, "f") == 0) {
      std::string vertex1, vertex2, vertex3;
      unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
      int matches = fscanf(file.get(), "%d/%d/%d %d/%d/%d %d/%d/%d\n",
                           &vertexIndex[0], &uvIndex[0], &normalIndex[0],
                           &vertexIndex[1], &uvIndex[1], &normalIndex[1],
                           &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
      if (matches != 9) {
        printf(
            "File can't be read by our simple parser :-( Try exporting with "
            "other options\n");
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
      if (fgets(stupidBuffer, 1000, file.get()) != stupidBuffer)
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
  return true;
}
}  // namespace Sculptor
