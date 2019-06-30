#ifndef RAYH
#define RAYH


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
		
        float3 A, B;
};		

#endif
