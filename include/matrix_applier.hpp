#pragma once

#include <vector>

#include "glm/glm.hpp"

namespace MatrixApplier {
void Apply(std::vector<glm::vec3>* vectors, glm::mat4 matrix);
}
