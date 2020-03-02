// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <memory>

namespace Sculptor {
class ShaderProgramBase;

class ShaderFactory {
 public:
  virtual ~ShaderFactory() = default;

  enum class ShaderType {
    PHONG,
    GOURAND,
  };
  enum class ObjectType {
    NORMAL,
    INSTANCED,
  };

  virtual std::unique_ptr<ShaderProgramBase> GetShader(
      ShaderType shader_type,
      ObjectType object_type) = 0;
};
}  // namespace Sculptor
