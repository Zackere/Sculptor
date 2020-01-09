cmake_minimum_required(VERSION 3.12)
project(Sculptor LANGUAGES CXX CUDA)

file(GLOB source_files  ${CMAKE_SOURCE_DIR}/src/*.cu ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/include/*.hpp)
add_executable(${PROJECT_NAME} main.cpp ${source_files})

# OpenGL
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw)

# Compile options
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS " -O3 --maxrregcount 32 -lineinfo")
set(CMAKE_CXX_FLAGS
    "-std=c++17                \
    -Weffc++                   \
    -Wredundant-decls          \
    -Wmissing-declarations     \
    -Wall                      \
    -Wextra                    \
    -Wshadow                   \
    -O2                        \
    -Wswitch-enum              \
    -Werror                    \
    -Wredundant-decls          \
    -Wformat=2                 \
    -Wmissing-format-attribute \
    -Wformat-nonliteral        \
    -g")
