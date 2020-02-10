// Copyright 2020 Wojciech Replin. All rights reserved.

#include "sculptor.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
#include <utility>

#include "../camera/basic_camera.hpp"
#include "../drill/drill.hpp"
#include "../glObject/gl_object.hpp"
#include "../kdtree_constructor/kdtree_cpu_std_constructor.hpp"
#include "../kdtree_remover/kdtree_gpu_remover.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../modelProvider/obj_provider.hpp"
#include "../sculptingMaterial/cube_sculpting_material.hpp"
#include "../shaderProvider/shader_provider.hpp"
#include "../textureProvider/png_texture_provider.hpp"

namespace Sculptor {
namespace {
struct {
  float width = 2 * 1280.f, height = 2 * 960.f;
  float aspect = width / height;
} window_properties;
void OnResize(GLFWwindow*, int width, int height) {
  glViewport(0, 0, window_properties.width = width,
             window_properties.height = height);
  window_properties.aspect = window_properties.width / window_properties.height;
}

bool main_running = false;
}  // namespace
Sculptor::Sculptor() {
  glfwInit();
}

Sculptor::~Sculptor() {
  glfwTerminate();
}

int Sculptor::Main() {
  if (main_running)
    return -1;
  main_running = true;

  GLFWwindow* window;
  window = glfwCreateWindow(window_properties.width, window_properties.height,
                            "Sculptor", nullptr, nullptr);
  if (!window)
    return -1;

  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK)
    return -1;
  glfwSetWindowSizeCallback(window, OnResize);

  glViewport(0, 0, window_properties.width, window_properties.height);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  glClearColor(44.0f / 255.0f, 219.0f / 255.0f, 216.0f / 255.0f, 0.0f);
  std::string base = "../Sculptor/";
  constexpr int ncubes = 100;
  std::unique_ptr<glObject> cube = std::make_unique<glObject>(
      std::make_unique<ObjProvider>(base + "model/cube.obj"),
      std::make_unique<ShaderProvider>(
          base + "shader/instancedCube/cube_shader.vs",
          base + "shader/cube/cube_shader.fs"),
      std::make_unique<MatrixApplier>(),
      std::make_unique<PNGTextureProvider>(base + "texture/cube.png"));
  cube->Transform(glm::scale(
      glm::mat4(1.f), glm::vec3(1.f / ncubes, 1.f / ncubes, 1.f / ncubes)));

  CubeSculptingMaterial material(
      ncubes, std::move(cube), std::make_unique<MatrixApplier>(),
      std::make_unique<KdTreeConstructor>(
          std::make_unique<KdTreeCPUStdConstructor>()),
      std::make_unique<KdTreeRemover>(
          std::make_unique<KdTreeGPURemoverHeurestic>()));

  std::unique_ptr<glObject> drill_model = std::make_unique<glObject>(
      std::make_unique<ObjProvider>(base + "model/drill.obj"),
      std::make_unique<ShaderProvider>(base + "shader/drill/drill_shader.vs",
                                       base + "shader/drill/drill_shader.fs"),
      std::make_unique<MatrixApplier>(), nullptr);
  drill_model->Transform(
      glm::scale(glm::mat4(1.f), glm::vec3(0.02, 0.02, 0.02)));
  Drill drill(std::move(drill_model));

  BasicCamera basic_camera({3, 1.5, 3}, {0, 0, 0}, {0, 1, 0});
  Camera* active_camera = &basic_camera;

  double old_mouse_pos_x, old_mouse_pos_y, cur_mouse_pos_x, cur_mouse_pos_y;
  glfwGetCursorPos(window, &cur_mouse_pos_x, &cur_mouse_pos_y);
  old_mouse_pos_x = cur_mouse_pos_x;
  old_mouse_pos_y = cur_mouse_pos_y;

  do {
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
      material.Rotate(-0.01f);
    else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
      material.Rotate(0.01f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
      drill.MoveForward();
    else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
      drill.MoveBackward();
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
      drill.MoveUp();
    else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
      drill.MoveDown();

    glfwGetCursorPos(window, &cur_mouse_pos_x, &cur_mouse_pos_y);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
      active_camera->Rotate({cur_mouse_pos_x - old_mouse_pos_x,
                             cur_mouse_pos_y - old_mouse_pos_y});
    old_mouse_pos_x = cur_mouse_pos_x;
    old_mouse_pos_y = cur_mouse_pos_y;

    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f), window_properties.aspect, 0.1f, 10.0f);
    auto vp = projection * basic_camera.GetTransform();

    drill.Spin();
    material.Collide(drill.GetObject());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    material.Render(vp);
    drill.Render(vp);
    glfwSwapBuffers(window);
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0);

  main_running = false;
  return 0;
}
}  // namespace Sculptor
