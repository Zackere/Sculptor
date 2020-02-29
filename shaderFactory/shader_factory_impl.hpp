// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <memory>

#include "shader_factory.hpp"

namespace Sculptor {
class ShaderProgramBase;

class ShaderFactoryImpl : public ShaderFactory {
 public:
  ~ShaderFactoryImpl() override = default;

  std::unique_ptr<ShaderProgramBase> GetShader(ShaderType shader_type,
                                               ObjectType object_type) override;
};
}  // namespace Sculptor
