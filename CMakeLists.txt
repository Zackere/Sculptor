cmake_minimum_required(VERSION 3.12)
project(Sculptor LANGUAGES CXX)
set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,cert-*,cppcoreguidelines-*,clang-analyzer-*,performance-*,modernize-*,bugprone-*)
set(CMAKE_CXX_CPPLINT "cpplint")

set(CMAKE_COMPILER_IS_GNUCC TRUE)
file(GLOB source_files
    ${CMAKE_SOURCE_DIR}/*/*.cu
    ${CMAKE_SOURCE_DIR}/*/*.cpp
    ${CMAKE_SOURCE_DIR}/*/*.hpp
    ${CMAKE_SOURCE_DIR}/*/*/*.hpp
    ${CMAKE_SOURCE_DIR}/*/*/*.cpp
    )
add_executable(${PROJECT_NAME} main.cpp ${source_files})
add_definitions(-D__INTELLISENSE__)

# Intel Thread Building Blocks
target_link_libraries(${PROJECT_NAME} tbb)

# Cuda
set(CMAKE_CUDA_HOST_COMPILER g++-8)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
include_directories($ENV{CUDA_PATH}/include)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(CUDA REQUIRED)
enable_language(CUDA)

# OpenGL
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw)

# Compile options
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 14)
set(CUDA_NVCC_FLAGS "--maxrregcount 32 -lineinfo")
set(CMAKE_CXX_FLAGS
    "-std=c++17 -O2 -g         \
    -Weffc++                   \
    -Wredundant-decls          \
    -Wall                      \
    -Wextra                    \
    -Wshadow                   \
    -Wswitch-enum              \
    -Werror                    \
    -Wredundant-decls          \
    -Wformat=2                 \
    -Wmissing-format-attribute \
    -Wformat-nonliteral")
