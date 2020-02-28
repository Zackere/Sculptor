// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <map>
#include <set>
#include <string>

namespace Sculptor {
class ShaderProgramBase;

class LightBase {
 public:
  LightBase(glm::vec3 ambient,
            glm::vec3 diffuse,
            glm::vec3 specular,
            std::string light_class_name);

  virtual ~LightBase() = 0;
  virtual void LoadIntoShader(ShaderProgramBase* shader);
  virtual void UnloadFromShader(ShaderProgramBase* shader);
  void Enable(ShaderProgramBase* shader);
  void Disable(ShaderProgramBase* shader);

  glm::vec3 GetAmbient() const { return ambient_; }
  glm::vec3 GetDiffuse() const { return diffuse_; }
  glm::vec3 GetSpecular() const { return specular_; }

  virtual void SetPosition(glm::vec3 pos) = 0;

 protected:
  unsigned GetId() const { return id_; }

 private:
  static std::map<std::string, std::set<unsigned>> taken_ids_;

  std::string light_class_name_;
  unsigned id_ = 0;

  glm::vec3 ambient_;
  glm::vec3 diffuse_;
  glm::vec3 specular_;
};
}  // namespace Sculptor
