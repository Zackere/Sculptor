#include "sculptor.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
#include <utility>

#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../modelProvider/obj_provider.hpp"
#include "../shaderProvider/shader_provider.hpp"
#include "../shapeGenerator/hollow_cube_generator.hpp"
#include "../textureProvider/png_texture_provider.hpp"

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

  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), static_cast<float>(wWidth) / wHeight, 0.1f, 100.0f);
  glm::mat4 view =
      glm::lookAt(glm::vec3(4, 2, 4), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  auto vp = projection * view;

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  glClearColor(44.0f / 255.0f, 219.0f / 255.0f, 216.0f / 255.0f, 0.0f);
  std::string base = "../Sculptor2/Sculptor/";
  constexpr int ncubes = 200;
  std::unique_ptr<glObject> cube = std::make_unique<glObject>(
      std::make_unique<ObjProvider>(base + "model/cube.obj"),
      std::make_unique<ShaderProvider>(
          base + "shader/instancedCube/cube_shader.vs",
          base + "shader/cube/cube_shader.fs"),
      std::make_unique<MatrixApplier>(),
      std::make_unique<PNGTextureProvider>(base + "texture/cube.png"));
  cube->Transform(glm::scale(
      glm::mat4(1.f), glm::vec3(1.f / ncubes, 1.f / ncubes, 1.f / ncubes)));

  glInstancedObject gi(ncubes, std::move(cube),
                       std::make_unique<HollowCubeGenerator>(2.f / ncubes),
                       std::make_unique<MatrixApplier>());

  do {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    gi.Render(vp);

    glfwSwapBuffers(window);

    gi.Transform(glm::rotate(glm::mat4(1.f), 0.01f, glm::vec3(0, 1, 0)));

    glfwPollEvents();
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0);
  return 0;
}
}  // namespace Sculptor
