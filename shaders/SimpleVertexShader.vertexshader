#version 330 core
layout(location = 0)in vec3 vertexPosition_modelspace;
layout(location = 1)in vec2 vertexUV;
layout(location = 2)in vec3 normal_vector;

in vec3 offset;

uniform mat4 mvp;

out vec2 UV;

void main(){
    gl_Position = mvp * vec4(vertexPosition_modelspace + offset, 1);
    UV = vertexUV;
}

