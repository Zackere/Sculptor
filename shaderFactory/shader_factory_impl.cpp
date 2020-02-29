// Copyright 2020 Wojciech Replin. All rights reserved.

#include "shader_factory_impl.hpp"

#include "../shaderProgram/shader_program.hpp"

namespace Sculptor {

std::unique_ptr<ShaderProgramBase> ShaderFactoryImpl::GetShader(
    ShaderFactory::ShaderType shader_type,
    ShaderFactory::ObjectType object_type) {
  std::string vertex_name = "vertex_shader.vs",
              fragment_name = "fragment_shader.fs",
              path = "../Sculptor/shader/";
  switch (shader_type) {
    case ShaderType::PHONG:
      vertex_name = "phong_" + vertex_name;
      fragment_name = "phong_" + fragment_name;
      path = path + "phong/";
      break;
  }
  switch (object_type) {
    case ObjectType::NORMAL:
      break;
    case ObjectType::INSTANCED:
      vertex_name = "instanced_" + vertex_name;
      break;
  }
  return std::make_unique<ShaderProgram>(path + vertex_name,
                                         path + fragment_name);
}

}  // namespace Sculptor
