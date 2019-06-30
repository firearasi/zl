#ifndef float3_H
#define float3_H
__device__ __host__ float3 cross(const float3 &v1, const float3 &v2);

__device__ __host__ float3 unit_vector(float3 v);

__device__ __host__ float3 lerp(const float3& u, const float3& v, float t);

__device__ __host__ float3 operator+(const float3 &v1, const float3 &v2);

__device__ __host__ float3 operator-(const float3 &v1, const float3 &v2);

__device__ __host__ float3 operator*(float t, const float3 &v);
#endif // float3_H
