#version 330 core
layout(location = 0)in vec3 vertex_position;
layout(location = 1)in vec2 vertex_uv;
layout(location = 2)in vec3 vertex_normal;
layout(location = 3)in mat4 model_transform;
layout(location = 7)in mat4 i_model_transform;

uniform mat4 vp;
uniform mat4 global_transform;
uniform mat4 i_global_transform;

out vec3 pos;
out vec3 normal;
out vec2 uv;

void main(){
    pos = vec3(global_transform * model_transform * vec4(vertex_position, 1));
    normal = vec3(transpose(i_model_transform * i_global_transform) * vec4(vertex_normal, 1));
    uv = vertex_uv;
    gl_Position = vp * vec4(pos, 1);
}

