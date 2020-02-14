#version 330 core
layout(location = 0)in vec3 vertex_position;
layout(location = 1)in vec2 vertex_uv;
layout(location = 2)in vec3 vertex_normal;

in float offset_x;
in float offset_y;
in float offset_z;

uniform mat4 vp;

out vec3 pos;
out vec3 normal;
out vec2 uv;

void main(){
    gl_Position = vp * vec4(vertex_position + vec3(offset_x, offset_y, offset_z), 1);
    pos = vertex_position;
    normal = vertex_normal;
    uv = vertex_uv;
}

