#version 330 core

uniform sampler2D texture_sampler;

in vec2 uv;
in vec3 light_color;

out vec3 color;

void main(){
    color = clamp(texture(texture_sampler, uv).rgb * light_color, 0.0, 1.0);
}

