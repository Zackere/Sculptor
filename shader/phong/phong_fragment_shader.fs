#version 330 core

struct DirectionalLight{
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool enabled;
};
#define NDIRECTIONALLIGHTS 1
uniform DirectionalLight SculptorDirectionalLight[NDIRECTIONALLIGHTS];

uniform vec4 light_coefficient;
uniform vec3 eye_pos;
uniform sampler2D texture_sampler;

in vec3 pos;
in vec3 normal;
in vec2 uv;

out vec3 color;

vec3 CalculateDirectionalLightContribution(vec3 normal, vec3 eye_dir);

void main(){
    vec3 n = normalize(normal);
    vec3 eye_dir = normalize(eye_pos - pos);

    color = vec3(0, 0, 0);

    color += CalculateDirectionalLightContribution(n, eye_dir);

    color *= texture(texture_sampler, uv).rgb;
    clamp(color, 0.0, 1.0);
}

vec3 CalculateDirectionalLightContribution(vec3 normal, vec3 eye_dir) {
    vec3 ret = vec3(0, 0, 0);
    for(int i = 0; i < NDIRECTIONALLIGHTS; ++i) {
        if(!SculptorDirectionalLight[i].enabled)
            continue;

        vec3 light = normalize(SculptorDirectionalLight[i].direction);
        float diff_cos = max(dot(light, normal), 0.0);
        float spec_cos = pow(max(dot(2 * diff_cos * normal - light, eye_dir), 0.0), light_coefficient.w);

        ret += light_coefficient.x * SculptorDirectionalLight[i].ambient
             + light_coefficient.y * SculptorDirectionalLight[i].diffuse * diff_cos
             + light_coefficient.z * SculptorDirectionalLight[i].specular * (diff_cos > 0 ? spec_cos : 0);
    }
    return ret;
}
