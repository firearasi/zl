#include "plane.h"

__device__ __host__ plane::plane(float3 _pt, float3 _normal)
{
    P=_pt;
    N=unit_vector(_normal);
}

__device__ __host__ float plane::distance_to_pt(float3 Q)
{

    return dot(Q-P,N);
}

 __device__ __host__ bool plane::intersects_with(const ray& r, float3& at)
 {
     return true;
 }
