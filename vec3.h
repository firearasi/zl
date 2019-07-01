#ifndef float3_H
#define float3_H
#include <math.h>
inline __device__ __host__ float3 unit_vector(const float3 & v) {
   float length = sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);
    return make_float3(v.x/length,v.y/length,v.z/length);
}

inline __device__ __host__ float dot(const float3 &v1, const float3 &v2) {
    return v1.x *v2.x + v1.y *v2.y  + v1.z *v2.z;
}

inline __device__ __host__ float3 cross(const float3 &v1, const float3 &v2) {
    return make_float3( (v1.y*v2.z - v1.z*v2.y),
                (-(v1.x*v2.z - v1.z*v2.x)),
                (v1.x*v2.y - v1.y*v2.x));
}

inline __device__ __host__ float3 operator+(const float3 &v1, const float3 &v2) {
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __device__ __host__ float3 operator-(const float3 &v1, const float3 &v2) {
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}



inline __device__ __host__ float3 operator*(float t, const float3 &v) {
    return make_float3(t*v.x, t*v.y, t*v.z);
}


inline __device__ __host__ float3 lerp(const float3& a, const float3& b, float t)
{
    return (1-t)*a+t*b;
}

inline __device__ __host__ float length(const float3& v)
{
    return sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);
}

inline __device__ __host__ float3 minus(const float3& v1, const float3& v2) {
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

#endif // float3_H
