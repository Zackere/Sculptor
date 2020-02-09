// Copyright 2020 Wojciech Replin. All rights reserved.

#include "shader_provider.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace Sculptor {

ShaderProvider::ShaderProvider(std::string_view vertex_shader_path,
                               std::string_view fragment_shader_path)
    : vertex_shader_path_(vertex_shader_path),
      fragment_shader_path_(fragment_shader_path) {}

GLuint ShaderProvider::Get() {
  // http://www.opengl-tutorial.org/beginners-tutorials/tutorial-2-the-first-triangle/
  // Create the shaders
  GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
  GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

  // Read the Vertex Shader code from the file
  std::string VertexShaderCode;
  std::ifstream VertexShaderStream(vertex_shader_path_.data(), std::ios::in);
  if (VertexShaderStream.is_open()) {
    std::stringstream sstr;
    sstr << VertexShaderStream.rdbuf();
    VertexShaderCode = sstr.str();
    VertexShaderStream.close();
  } else {
    std::cerr << "Impossible to open " << vertex_shader_path_
              << ". Are you in the right directory ? Don't "
                 "forget to read the FAQ !\n";
    return 0;
  }

  // Read the Fragment Shader code from the file
  std::string FragmentShaderCode;
  std::ifstream FragmentShaderStream(fragment_shader_path_.data(),
                                     std::ios::in);
  if (FragmentShaderStream.is_open()) {
    std::stringstream sstr;
    sstr << FragmentShaderStream.rdbuf();
    FragmentShaderCode = sstr.str();
    FragmentShaderStream.close();
  }

  GLint Result = GL_FALSE;
  int InfoLogLength;

  // Compile Vertex Shader
  std::cout << "Compiling shader : " << vertex_shader_path_ << "\n";
  char const* VertexSourcePointer = VertexShaderCode.c_str();
  glShaderSource(VertexShaderID, 1, &VertexSourcePointer, nullptr);
  glCompileShader(VertexShaderID);

  // Check Vertex Shader
  glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, nullptr,
                       &VertexShaderErrorMessage[0]);
    std::cout << VertexShaderErrorMessage.data() << std::endl;
  }

  // Compile Fragment Shader
  std::cout << "Compiling shader : " << fragment_shader_path_ << "\n";
  char const* FragmentSourcePointer = FragmentShaderCode.c_str();
  glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, nullptr);
  glCompileShader(FragmentShaderID);

  // Check Fragment Shader
  glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
    glGetShaderInfoLog(FragmentShaderID, InfoLogLength, nullptr,
                       &FragmentShaderErrorMessage[0]);
    std::cout << FragmentShaderErrorMessage.data() << std::endl;
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
    glGetProgramInfoLog(ProgramID, InfoLogLength, nullptr,
                        &ProgramErrorMessage[0]);
    std::cout << ProgramErrorMessage.data() << std::endl;
  }

  glDetachShader(ProgramID, VertexShaderID);
  glDetachShader(ProgramID, FragmentShaderID);

  glDeleteShader(VertexShaderID);
  glDeleteShader(FragmentShaderID);

  return ProgramID;
}
}  // namespace Sculptor
