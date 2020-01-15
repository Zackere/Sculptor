#include "../include/sculptor.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "../external/lodepng/lodepng.cpp"
#include "../include/sculpting_material.hpp"
#include "../include/shader_loader.hpp"
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Sculptor {
Sculptor::Sculptor() {
  glfwInit();
}

Sculptor::~Sculptor() {
  glfwTerminate();
}

int Sculptor::Main() {
  GLFWwindow* window;
  constexpr float wWidth = 1280.f, wHeight = 960.f;
  window = glfwCreateWindow(static_cast<int>(wWidth), static_cast<int>(wHeight),
                            "Sculptor", nullptr, nullptr);
  if (!window)
    return -1;

  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK)
    return -1;

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  GLuint VertexArrayID;
  glGenVertexArrays(1, &VertexArrayID);
  glBindVertexArray(VertexArrayID);

  constexpr float side_len = 5;
  SculptingMaterial material(SculptingMaterial::MaterialType::CUBE,
                             SculptingMaterial::InitialShape::CUBE, side_len);

  std::vector<unsigned char> image;  // the raw pixels
  unsigned width, height;
  unsigned error = lodepng::decode(image, width, height,
                                   "../Sculptor/models/CubeTexture.png");
  if (error)
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;

  GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, image.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  auto programID = ShaderLoader::Load(
      "../Sculptor/shaders/SimpleVertexShader.vertexshader",
      "../Sculptor/shaders/SimpleFragmentShader.fragmentshader");

  glm::mat4 Projection = glm::perspective(
      glm::radians(45.0f), static_cast<float>(wWidth) / wHeight, 0.1f, 100.0f);
  glm::mat4 View =
      glm::lookAt(glm::vec3(4, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  auto vp = Projection * View;
  GLuint MatrixID = glGetUniformLocation(programID, "mvp");
  GLuint OffsetsID = glGetAttribLocation(programID, "offset");
  glBindBuffer(GL_ARRAY_BUFFER, material.GetMaterialElementsBuffer());
  glVertexAttribDivisor(OffsetsID, 1);

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  glClearColor(44.0f / 255.0f, 219.0f / 255.0f, 216.0f / 255.0f, 0.0f);

  material.RemoveAt(0);
  material.RemoveAt(3);
  material.RemoveAt(5);
  do {
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
      vp = glm::rotate(vp, -0.01f, glm::vec3(0, 1, 0));
    else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
      vp = glm::rotate(vp, 0.01f, glm::vec3(0, 1, 0));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, material.GetVerticiesBuffer());
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, material.GetUVBuffer());
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, material.GetNormalsBuffer());
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glEnableVertexAttribArray(OffsetsID);
    glBindBuffer(GL_ARRAY_BUFFER, material.GetMaterialElementsBuffer());
    glVertexAttribPointer(OffsetsID, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glUseProgram(programID);
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &vp[0][0]);
    glDrawArraysInstanced(GL_TRIANGLES, 0, material.GetNVertices(),
                          material.GetMaterialElements().size());

    glDisableVertexAttribArray(0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0);
  return 0;
}
}  // namespace Sculptor
