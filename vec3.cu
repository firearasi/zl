#include <math.h>
#include "vec3.h"


__device__ __host__ float3 unit_vector(float3 v) {
   float length = sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);
    return make_float3(v.x/length,v.y/length,v.z/length);
}

__device__ __host__ float dot(const float3 &v1, const float3 &v2) {
    return v1.x *v2.x + v1.y *v2.y  + v1.z *v2.z;
}

__device__ __host__ float3 cross(const float3 &v1, const float3 &v2) {
    return make_float3( (v1.y*v2.z - v1.z*v2.y),
                (-(v1.x*v2.z - v1.z*v2.x)),
                (v1.x*v2.y - v1.y*v2.x));
}

__device__ __host__ float3 operator+(const float3 &v1, const float3 &v2) {
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ __host__ float3 operator-(const float3 &v1, const float3 &v2) {
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}


__device__ __host__ float3 operator*(float t, const float3 &v) {
    return make_float3(t*v.x, t*v.y, t*v.z);
}


__device__ __host__ float3 lerp(const float3& a, const float3& b, float t)
{
    return (1-t)*a+t*b;
}
