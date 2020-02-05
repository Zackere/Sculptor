#version 330 core

in vec2 UV;
in vec3 normal;

out vec3 color;

uniform sampler2D textureSampler;

void main(){
    color = texture(textureSampler, UV).rgb;
}
