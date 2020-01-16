#pragma once

#include <vector>

#include "glm/glm.hpp"

namespace MatrixApplier {
void Apply(glm::vec3* vectors, int n, glm::mat4 matrix);
}
