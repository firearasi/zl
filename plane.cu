#include "plane.h"

__device__ __host__ plane::plane(float3 _pt, float3 _normal)
{
    P=_pt;
    N=unit_vector(_normal);
}

__device__ __host__ inline float plane::distance_to_pt(float3 Q)
{

    return dot(Q-P,N);
}

/*
 __device__ __host__ bool plane::intersects_with(const ray& r, float3& at)
 {
     float a=dot(r.direction(),N);
     float b=dot(P-r.origin(),N);
     if(a==0)
         return false;
     float t=b/a;
     if(t<=0)
         return false;
     at=r.point_at_parameter(t);
     return true;
 }
*/
