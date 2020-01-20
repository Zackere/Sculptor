#include "../include/sculptor.hpp"

#include "../include/drill.hpp"
#include "../include/kdtree_cpu.hpp"
#include "../include/sculpting_material.hpp"
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
  constexpr auto wWidth = 1280.f, wHeight = 960.f;
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

  constexpr float side_len = 60;
  SculptingMaterial material(SculptingMaterial::MaterialType::CUBE,
                             SculptingMaterial::InitialShape::CUBE, side_len,
                             std::make_unique<KdTreeCPU>());
  Drill drill;

  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), static_cast<float>(wWidth) / wHeight, 0.1f, 100.0f);
  glm::mat4 view =
      glm::lookAt(glm::vec3(3, 0, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  auto vp = projection * view;

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  glClearColor(44.0f / 255.0f, 219.0f / 255.0f, 216.0f / 255.0f, 0.0f);

  do {
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
      material.Rotate(-0.01f);
    else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
      material.Rotate(0.01f);
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
      drill.MoveForward();
    else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
      drill.MoveBackward();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    material.Collide(drill);
    drill.NextFrame();

    material.Render(vp);
    drill.Render(vp);

    glfwSwapBuffers(window);
    glfwPollEvents();
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0);
  return 0;
}
}  // namespace Sculptor
