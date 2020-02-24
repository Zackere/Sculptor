#version 330 core
layout(location = 0)in vec3 vertex_position;
layout(location = 1)in vec2 vertex_uv;
layout(location = 2)in vec3 vertex_normal;
layout(location = 3)in mat4 model_transform;

uniform mat4 vp;

out vec3 pos;
out vec3 normal;
out vec2 uv;

void main(){
    pos = vec3(model_transform * vec4(vertex_position, 1));
    normal = vertex_normal;
    uv = vertex_uv;
    gl_Position = vp * vec4(pos, 1);
}

