#include "aabb.h"


__device__ __host__ bool aabb::contains(float3 pt)
{
    return (_min.x <= pt.x && pt.x <= _max.x) &&
           (_min.y <= pt.y && pt.y <= _max.y) &&
           (_min.z <= pt.z && pt.z <= _max.z);
}

int aabb_mesh::count_density_for_point_cloud(std::vector<float3>* pc, int i, int j, int k)
{
    float3 lower,upper;
    lower.x=box.min().x+(box.max().x-box.min().x)/m*i;
    upper.x=box.min().x+(box.max().x-box.min().x)/m*(i+1);
    lower.y=box.min().y+(box.max().y-box.min().y)/n*j;
    upper.y=box.min().y+(box.max().y-box.min().y)/n*(j+1);
    lower.z=box.min().z+(box.max().z-box.min().z)/p*k;
    upper.z=box.min().z+(box.max().z-box.min().z)/p*(k+1);
    aabb subbox(lower,upper);
    int count = 0;
    for(float3 pt:*pc)
    {
        if(subbox.contains(pt))
            count++;
    }
    return count;
}
