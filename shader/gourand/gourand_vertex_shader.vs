#version 330 core

layout(location = 0)in vec3 vertex_position;
layout(location = 1)in vec2 vertex_uv;
layout(location = 2)in vec3 vertex_normal;

struct DirectionalLight{
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool enabled;
};
#define NDIRECTIONALLIGHTS 2
uniform DirectionalLight SculptorDirectionalLight[NDIRECTIONALLIGHTS];

struct PointLight{
    vec3 position;
    vec3 attenuation;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool enabled;
};
#define NPOINTLIGHTS 4
uniform PointLight SculptorPointLight[NPOINTLIGHTS];

struct Spotlight{
    vec3 position;
    vec3 look_target;
    vec2 cutoff;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool enabled;
};
#define NSPOTLIGHTS 2
uniform Spotlight SculptorSpotlight[NSPOTLIGHTS];

uniform vec4 light_coefficient;
uniform vec3 eye_pos;
uniform mat4 vp;
uniform bool blinn;

out vec2 uv;
out vec3 light_color;

vec3 CalculateDirectionalLightContribution(vec3 pos, vec3 normal, vec3 eye_dir);
vec3 CalculatePointLightContribution(vec3 pos, vec3 normal, vec3 eye_dir);
vec3 CalculateSpotlightContribution(vec3 pos, vec3 normal, vec3 eye_dir);

void main(){
    uv = vertex_uv;
    gl_Position = vp * vec4(vertex_position, 1);

    vec3 normal = normalize(vertex_normal);
    vec3 eye_dir = normalize(eye_pos - vertex_position);
    light_color = CalculateDirectionalLightContribution(vertex_position, normal, eye_dir) +
                  CalculatePointLightContribution(vertex_position, normal, eye_dir) +
                  CalculateSpotlightContribution(vertex_position, normal, eye_dir);
}

vec3 CalculateDirectionalLightContribution(vec3 pos, vec3 normal, vec3 eye_dir) {
    vec3 ret = vec3(0, 0, 0);
    for(int i = 0; i < NDIRECTIONALLIGHTS; ++i) {
        if(!SculptorDirectionalLight[i].enabled)
            continue;

        vec3 light = normalize(SculptorDirectionalLight[i].direction);
        float diff_cos = max(dot(light, normal), 0.0);
        float spec_cos = 0.0;
        if(!blinn)
            spec_cos = pow(max(dot(2 * diff_cos * normal - light, eye_dir), 0.0), light_coefficient.w);
        else
            spec_cos = pow(max(dot(normal, normalize(light + eye_dir)), 0.0), 4.0 * light_coefficient.w);

        ret += light_coefficient.x * SculptorDirectionalLight[i].ambient
             + light_coefficient.y * SculptorDirectionalLight[i].diffuse * diff_cos
             + light_coefficient.z * SculptorDirectionalLight[i].specular * (diff_cos > 0 ? spec_cos : 0.0);
    }
    return ret;
}

vec3 CalculatePointLightContribution(vec3 pos, vec3 normal, vec3 eye_dir) {
    vec3 ret = vec3(0, 0, 0);
    for(int i = 0; i < NPOINTLIGHTS; ++i) {
        if(!SculptorPointLight[i].enabled)
            continue;

        float distance = length(SculptorPointLight[i].position - pos);
        float att = 1.0 / dot(vec3(1, distance, distance * distance), SculptorPointLight[i].attenuation);

        vec3 light = (SculptorPointLight[i].position - pos) / distance;
        float diff_cos = max(dot(light, normal), 0.0);
        float spec_cos = 0.0;
        if(!blinn)
            spec_cos = pow(max(dot(2 * diff_cos * normal - light, eye_dir), 0.0), light_coefficient.w);
        else
            spec_cos = pow(max(dot(normal, normalize(light + eye_dir)), 0.0), 4.0 * light_coefficient.w);

        ret += (light_coefficient.x * SculptorPointLight[i].ambient
              + light_coefficient.y * SculptorPointLight[i].diffuse * diff_cos
              + light_coefficient.z * SculptorPointLight[i].specular * (diff_cos > 0 ? spec_cos : 0)
               ) * att;
    }
    return ret;
}

vec3 CalculateSpotlightContribution(vec3 pos, vec3 normal, vec3 eye_dir) {
    vec3 ret = vec3(0, 0, 0);
    for(int i = 0; i < NSPOTLIGHTS; ++i) {
        if(!SculptorSpotlight[i].enabled)
            continue;

        vec3 light = normalize(SculptorSpotlight[i].position - pos);
        float theta = dot(light, normalize(SculptorSpotlight[i].position - SculptorSpotlight[i].look_target));
        if(theta < SculptorSpotlight[i].cutoff.x)
            continue;
        float intensity = clamp((theta - SculptorSpotlight[i].cutoff.x) /
                          (SculptorSpotlight[i].cutoff.x - SculptorSpotlight[i].cutoff.y), 0.0, 1.0);
        float diff_cos = max(dot(light, normal), 0.0);
        float spec_cos = 0.0;
        if(!blinn)
            spec_cos = pow(max(dot(2 * diff_cos * normal - light, eye_dir), 0.0), light_coefficient.w);
        else
            spec_cos = pow(max(dot(normal, normalize(light + eye_dir)), 0.0), 4.0 * light_coefficient.w);

        ret += light_coefficient.x * SculptorSpotlight[i].ambient
             + (light_coefficient.y * SculptorSpotlight[i].diffuse * diff_cos +
             + light_coefficient.z * SculptorSpotlight[i].specular * (diff_cos > 0 ? spec_cos : 0)) * intensity;
    }
    return ret;
}

