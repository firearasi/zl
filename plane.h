#ifndef PLANE_H
#define PLANE_H
#include "vec3.h"
#include "ray.h"

class plane
{
public:
    __device__ __host__ plane(float3 _pt, float3 _normal);
    __device__ __host__ float distance_to_pt(float3 aPt);
    __device__ __host__ bool intersects_with(const ray& r, float3& at);
    float3 P;
    float3 N;
};

#endif // PLANE_H
