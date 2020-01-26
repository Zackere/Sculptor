#version 330 core
layout(location = 0)in vec3 vertexPosition_modelspace;
layout(location = 1)in vec2 vertexUV;
layout(location = 2)in vec3 normal_vector;

in float offset_x;
in float offset_y;
in float offset_z;

uniform mat4 mvp;

out vec2 UV;

void main(){
    gl_Position = mvp * vec4(vertexPosition_modelspace + vec3(offset_x, offset_y, offset_z), 1);
    UV = vertexUV;
}

