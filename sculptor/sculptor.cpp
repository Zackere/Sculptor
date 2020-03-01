// Copyright 2020 Wojciech Replin. All rights reserved.

#include "sculptor.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../camera/basic_camera.hpp"
#include "../camera/follower_camera.hpp"
#include "../camera/third_person_camera.hpp"
#include "../drill/drill.hpp"
#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"
#include "../light/directional_light.hpp"
#include "../light/point_light.hpp"
#include "../light/spotlight.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../modelProvider/obj_provider.hpp"
#include "../sculptingMaterial/cube_sculpting_material_cpu.hpp"
#include "../shaderFactory/shader_factory_impl.hpp"
#include "../shaderProgram/shader_program.hpp"
#include "../textureProvider/png_texture_provider.hpp"

namespace Sculptor {
namespace {
bool main_running = false;
const glm::vec3 kUp{0.f, 1.f, 0.f};

struct {
  float width = 2 * 1280.f, height = 2 * 960.f;
  float aspect = width / height;
} window_properties;
void OnResize(GLFWwindow*, int width, int height) {
  glViewport(0, 0, window_properties.width = width,
             window_properties.height = height);
  window_properties.aspect = window_properties.width / window_properties.height;
}

std::set<int> keys_pressed = {};
void OnKeyPressed(GLFWwindow*, int key, int, int action, int) {
  if (action == GLFW_PRESS)
    keys_pressed.insert(key);
}
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
  glfwSetKeyCallback(window, OnKeyPressed);
  const auto is_key_pressed = [window](int key) {
    return glfwGetKey(window, key) == GLFW_PRESS;
  };

  glViewport(0, 0, window_properties.width, window_properties.height);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  glClearColor(44.0f / 255.0f, 219.0f / 255.0f, 216.0f / 255.0f, 0.0f);
  std::string base = "../Sculptor/";
  constexpr int ncubes = 6;

  std::unique_ptr<ShaderFactory> shader_factory =
      std::make_unique<ShaderFactoryImpl>();

  std::unique_ptr<SculptingMaterial> material =
      std::make_unique<CubeSculptingMaterialCPU>(
          ncubes, shader_factory.get(), std::make_unique<MatrixApplier>());

  std::unique_ptr<glObject> drill_model = std::make_unique<glObject>(
      std::make_unique<ObjProvider>(base + "model/cube.obj"),
      shader_factory->GetShader(ShaderFactory::ShaderType::PHONG,
                                ShaderFactory::ObjectType::NORMAL),
      std::make_unique<MatrixApplier>(),
      std::make_unique<PNGTextureProvider>(base + "texture/cube.png"),
      glm::vec4{1.0, 0.4, 1.0, 10.0});
  drill_model->Transform(
      glm::scale(glm::mat4(1.f), glm::vec3(0.02, 0.02, 0.02)));
  Drill drill(std::move(drill_model));

  BasicCamera static_camera({3, 1.5, 3}, {0, 0, 0}, kUp);
  FollowerCamera follower_camera({3, 1.5, 3}, &drill.GetObject(), kUp);
  ThirdPersonCamera thrid_person_camera({0.6, 0.2, 0}, &drill.GetObject(), kUp);
  Camera* active_camera = &static_camera;

  std::vector<std::unique_ptr<LightBase>> day_lights;
  day_lights.push_back(std::make_unique<DirectionalLight>(
      glm::vec3{0.3, 0.3, 0.3}, glm::vec3{1.0, 1.0, 1.0},
      glm::vec3{1.0, 1.0, 1.0}, glm::vec3{3.0, 3.0, 3.0}));
  day_lights.push_back(std::make_unique<Spotlight>(
      glm::vec3{0.0, 0.0, 0.0}, glm::vec3{5.0, 5.0, 5.0},
      glm::vec3{5.0, 5.0, 5.0}, glm::vec3{1.2, 1.2, 1.2},
      glm::vec3{0.0, 0.0, 0.0}, glm::vec2{0.7, 0.6}));
  day_lights.push_back(std::make_unique<PointLight>(
      glm::vec3{0.0, 0.0, 0.0}, glm::vec3{5.0, 5.0, 5.0},
      glm::vec3{5.0, 5.0, 5.0}, glm::vec3{0.0, -2.0, 0.0},
      glm::vec3{1.0, 1.5, 2.0}));
  std::vector<std::unique_ptr<LightBase>> night_lights;
  night_lights.push_back(std::make_unique<DirectionalLight>(
      glm::vec3{0.0, 0.0, 0.0}, glm::vec3{1.0, 1.0, 1.0},
      glm::vec3{1.0, 1.0, 1.0}, glm::vec3{3.0, 3.0, 3.0}));
  night_lights.push_back(std::make_unique<Spotlight>(
      glm::vec3{0.0, 0.0, 0.0}, glm::vec3{5.0, 5.0, 5.0},
      glm::vec3{5.0, 5.0, 5.0}, glm::vec3{1.2, 1.2, 1.2},
      glm::vec3{0.0, 0.0, 0.0}, glm::vec2{0.7, 0.6}));
  night_lights.push_back(std::make_unique<PointLight>(
      glm::vec3{0.0, 0.0, 0.0}, glm::vec3{5.0, 5.0, 5.0},
      glm::vec3{5.0, 5.0, 5.0}, glm::vec3{0.0, -2.0, 0.0},
      glm::vec3{1.0, 1.5, 2.0}));
  std::vector<std::unique_ptr<LightBase>> const* active_lights = &day_lights;

  double old_mouse_pos_x, old_mouse_pos_y, cur_mouse_pos_x, cur_mouse_pos_y;
  glfwGetCursorPos(window, &cur_mouse_pos_x, &cur_mouse_pos_y);
  old_mouse_pos_x = cur_mouse_pos_x;
  old_mouse_pos_y = cur_mouse_pos_y;

  do {
    glfwPollEvents();

    for (auto k : keys_pressed) {
      switch (k) {
        case GLFW_KEY_N:
          for (auto const& light : *active_lights) {
            light->Disable(drill.GetObject().GetShader());
            material->GetObject().Unload(light.get());
          }
          if (active_lights == &day_lights) {
            active_lights = &night_lights;
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
          } else if (active_lights == &night_lights) {
            active_lights = &day_lights;
            glClearColor(44.0f / 255.0f, 219.0f / 255.0f, 216.0f / 255.0f,
                         0.0f);
          }
          break;
        case GLFW_KEY_B:
          drill.GetObject().SetShader(
              shader_factory->GetShader(ShaderFactory::ShaderType::BLINN,
                                        ShaderFactory::ObjectType::NORMAL));
          material->GetObject().SetShader(
              shader_factory->GetShader(ShaderFactory::ShaderType::BLINN,
                                        ShaderFactory::ObjectType::INSTANCED));
          break;
        case GLFW_KEY_P:
          drill.GetObject().SetShader(
              shader_factory->GetShader(ShaderFactory::ShaderType::PHONG,
                                        ShaderFactory::ObjectType::NORMAL));
          material->GetObject().SetShader(
              shader_factory->GetShader(ShaderFactory::ShaderType::PHONG,
                                        ShaderFactory::ObjectType::INSTANCED));
          break;
        case GLFW_KEY_1:
          active_camera = &static_camera;
          break;
        case GLFW_KEY_2:
          active_camera = &follower_camera;
          break;
        case GLFW_KEY_3:
          active_camera = &thrid_person_camera;
          break;
      }
    }
    keys_pressed.clear();

    if (is_key_pressed(GLFW_KEY_LEFT))
      material->GetObject().Transform(glm::rotate(glm::mat4(1.f), -0.01f, kUp));
    else if (is_key_pressed(GLFW_KEY_RIGHT))
      material->GetObject().Transform(glm::rotate(glm::mat4(1.f), 0.01f, kUp));
    if (is_key_pressed(GLFW_KEY_DOWN))
      active_camera->Zoom(-0.1f);
    else if (is_key_pressed(GLFW_KEY_UP))
      active_camera->Zoom(0.1f);
    if (is_key_pressed(GLFW_KEY_W))
      drill.MoveForward();
    else if (is_key_pressed(GLFW_KEY_S))
      drill.MoveBackward();
    if (is_key_pressed(GLFW_KEY_E))
      drill.MoveUp();
    else if (is_key_pressed(GLFW_KEY_Q))
      drill.MoveDown();

    glfwGetCursorPos(window, &cur_mouse_pos_x, &cur_mouse_pos_y);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
      active_camera->Rotate({cur_mouse_pos_x - old_mouse_pos_x,
                             cur_mouse_pos_y - old_mouse_pos_y});
    old_mouse_pos_x = cur_mouse_pos_x;
    old_mouse_pos_y = cur_mouse_pos_y;

    drill.Spin();
    material->CollideWith(drill.GetObject());

    for (auto const& light : *active_lights) {
      drill.GetObject().Load(light.get());
      material->GetObject().Load(light.get());
    }
    drill.GetObject().Load(active_camera);
    material->GetObject().Load(active_camera);

    glm::mat4 const projection = glm::perspective(
        glm::radians(45.0f), window_properties.aspect, 0.1f, 10.0f);
    auto vp = projection * active_camera->GetTransform();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    material->GetObject().Render(vp);
    drill.Render(vp);

    glfwSwapBuffers(window);
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0);

  main_running = false;
  return 0;
}  // namespace Sculptor
}  // namespace Sculptor
