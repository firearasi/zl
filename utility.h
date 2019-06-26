#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include "aabb.h"

aabb point_cloud_bounds(const std::vector<float3>& pc);

#endif // UTILITY_H
