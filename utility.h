#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include "aabb.h"
#include "vec3.h"

aabb point_cloud_bounds(const std::vector<float3>& pc);

float3 heat_color(float value, float max_value);
#endif // UTILITY_H
