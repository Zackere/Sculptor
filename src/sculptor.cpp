#include "../include/sculptor.hpp"

#include "../include/drill.hpp"
#include "../include/kdtree.hpp"
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

  GLuint VertexArrayID;
  glGenVertexArrays(1, &VertexArrayID);
  glBindVertexArray(VertexArrayID);

  constexpr float side_len = 300;
  SculptingMaterial material(SculptingMaterial::MaterialType::CUBE,
                             SculptingMaterial::InitialShape::CUBE, side_len);
  Drill drill;
  KdTree collider;

  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), static_cast<float>(wWidth) / wHeight, 0.1f, 100.0f);
  glm::mat4 view =
      glm::lookAt(glm::vec3(3, 0, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  auto vp = projection * view;
  glBindBuffer(GL_ARRAY_BUFFER, material.GetMaterialElementsBuffer());

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  glClearColor(44.0f / 255.0f, 219.0f / 255.0f, 216.0f / 255.0f, 0.0f);

  for (int i = 0; i < 125; i += 2)
    material.RemoveAt(i);
  do {
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
      material.Rotate(-0.1f);
    else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
      material.Rotate(0.1f);
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
      drill.MoveForward();
    else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
      drill.MoveBackward();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    material.Enable();
    material.Render(vp);

    drill.NextFrame();
    drill.Enable();
    drill.Render(vp);

    glDisableVertexAttribArray(0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0);
  return 0;
}
}  // namespace Sculptor
