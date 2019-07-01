#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
	public:
        __device__ __host__ ray() {}
        __device__ __host__ ray(const float3& a, const float3 &b) {A=a; B=b;}
        __device__ __host__ float3 origin() const {return A;}
        __device__ __host__ float3 direction() const {return B;}
        __device__ __host__ float3 point_at_parameter(float t) const {
            return make_float3(A.x+t*B.x,A.y+t*B.y,A.z+t*B.z);
        }

        __device__ __host__ float distance_to_pt(float3 pt)
        {
            float3 vec = pt-origin();
            float length_vec = length(vec);
            float cos_theta = dot(unit_vector(vec),unit_vector(direction()));
            float sin_theta = sqrtf(1-cos_theta*cos_theta);
            return length_vec*sin_theta;
        }
		
        float3 A, B;
};		

#endif
