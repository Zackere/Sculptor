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
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Sculptor {
namespace {
GLuint LoadShaders(std::string_view vertex_file_path,
                   std::string_view fragment_file_path) {
  // http://www.opengl-tutorial.org/beginners-tutorials/tutorial-2-the-first-triangle/
  // Create the shaders
  GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
  GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

  // Read the Vertex Shader code from the file
  std::string VertexShaderCode;
  std::ifstream VertexShaderStream(vertex_file_path.data(), std::ios::in);
  if (VertexShaderStream.is_open()) {
    std::stringstream sstr;
    sstr << VertexShaderStream.rdbuf();
    VertexShaderCode = sstr.str();
    VertexShaderStream.close();
  } else {
    std::cerr << "Impossible to open " << vertex_file_path
              << ". Are you in the right directory ? Don't "
                 "forget to read the FAQ !\n";
    return 0;
  }

  // Read the Fragment Shader code from the file
  std::string FragmentShaderCode;
  std::ifstream FragmentShaderStream(fragment_file_path.data(), std::ios::in);
  if (FragmentShaderStream.is_open()) {
    std::stringstream sstr;
    sstr << FragmentShaderStream.rdbuf();
    FragmentShaderCode = sstr.str();
    FragmentShaderStream.close();
  }

  GLint Result = GL_FALSE;
  int InfoLogLength;

  // Compile Vertex Shader
  std::cout << "Compiling shader : " << vertex_file_path << "\n";
  char const* VertexSourcePointer = VertexShaderCode.c_str();
  glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
  glCompileShader(VertexShaderID);

  // Check Vertex Shader
  glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL,
                       &VertexShaderErrorMessage[0]);
    printf("%s\n", &VertexShaderErrorMessage[0]);
  }

  // Compile Fragment Shader
  std::cout << "Compiling shader : " << fragment_file_path << "\n";
  char const* FragmentSourcePointer = FragmentShaderCode.c_str();
  glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
  glCompileShader(FragmentShaderID);

  // Check Fragment Shader
  glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
    glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL,
                       &FragmentShaderErrorMessage[0]);
    printf("%s\n", &FragmentShaderErrorMessage[0]);
  }

  // Link the program
  std::cout << "Linking program.\n";
  GLuint ProgramID = glCreateProgram();
  glAttachShader(ProgramID, VertexShaderID);
  glAttachShader(ProgramID, FragmentShaderID);
  glLinkProgram(ProgramID);

  // Check the program
  glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
  glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
    glGetProgramInfoLog(ProgramID, InfoLogLength, NULL,
                        &ProgramErrorMessage[0]);
    printf("%s\n", &ProgramErrorMessage[0]);
  }

  glDetachShader(ProgramID, VertexShaderID);
  glDetachShader(ProgramID, FragmentShaderID);

  glDeleteShader(VertexShaderID);
  glDeleteShader(FragmentShaderID);

  return ProgramID;
}
}  // namespace
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

  constexpr float side_len = 20;
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

  auto programID =
      LoadShaders("../Sculptor/shaders/SimpleVertexShader.vertexshader",
                  "../Sculptor/shaders/SimpleFragmentShader.fragmentshader");

  glm::mat4 Projection = glm::perspective(
      glm::radians(45.0f), (float)wWidth / (float)wHeight, 0.1f, 100.0f);
  glm::mat4 View =
      glm::lookAt(glm::vec3(4, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  auto vp = Projection * View;
  GLuint MatrixID = glGetUniformLocation(programID, "mvp");

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  glClearColor(44.0f / 255.0f, 219.0f / 255.0f, 216.0f / 255.0f, 0.0f);

  do {
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
      vp = glm::rotate(vp, -0.01f, glm::vec3(0, 1, 0));
    else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
      vp = glm::rotate(vp, 0.01f, glm::vec3(0, 1, 0));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, material.GetVerticiesBuffer());
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glUnmapBuffer(GL_ARRAY_BUFFER);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, material.GetUVBuffer());
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, material.GetNormalsBuffer());
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glUseProgram(programID);
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &vp[0][0]);
    glDrawArrays(GL_TRIANGLES, 0, material.GetNVerticies());

    glDisableVertexAttribArray(0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0);
  return 0;
}
}  // namespace Sculptor
