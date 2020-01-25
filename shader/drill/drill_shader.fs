#version 330 core

in vec3 normal;

out vec3 color;

void main(){
    vec3 n = normalize(normal);
    vec3 light = normalize(vec3(1, -1, 1));
    float d = max(dot(light, n), 0.0);
	color = vec3(0.5, 0.5, 0.5) * d;
}

