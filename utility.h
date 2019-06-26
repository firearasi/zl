#ifndef UTILITY_H
#define UTILITY_H

#include <vector>

void point_cloud_bounds(const std::vector<float3>& pc, float2& x_bound, float2& y_bound, float2& z_bound);

#endif // UTILITY_H
