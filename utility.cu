#include "utility.h"
#include <limits>
#include <algorithm>

// 找出点云的上下界
aabb point_cloud_bounds(const std::vector<float3>& pc)
{
    float3 lower, upper;

    lower.x=std::numeric_limits<float>::max();
    upper.x=std::numeric_limits<float>::min();
    lower.y=std::numeric_limits<float>::max();
    upper.y=std::numeric_limits<float>::min();
    lower.z=std::numeric_limits<float>::max();
    upper.z=std::numeric_limits<float>::min();

    for(float3 pt: pc)
    {
        lower.x=std::min(lower.x,pt.x);
        upper.x=std::max(upper.x,pt.x);

        lower.y=std::min(lower.y,pt.y);
        upper.y=std::max(upper.y,pt.y);

        lower.z=std::min(lower.z,pt.z);
        upper.z=std::max(upper.z,pt.z);
    }

    return aabb(lower,upper);
}
