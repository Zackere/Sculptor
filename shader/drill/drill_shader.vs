#version 330 core
layout(location = 0)in vec3 vertexPosition_modelspace;
layout(location = 1)in vec2 vertexUV;
layout(location = 2)in vec3 normal_vector;

uniform mat4 mvp;

out vec3 pos;
out vec3 normal;

void main(){
    gl_Position = mvp * vec4(vertexPosition_modelspace, 1);
    pos = vertexPosition_modelspace;
    normal = normal_vector;
}
