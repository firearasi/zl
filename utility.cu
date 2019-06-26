#include "utility.h"
#include <limits>
#include <algorithm>

// 找出点云的上下界
void point_cloud_bounds(const std::vector<float3>& pc, float2& x_bound, float2& y_bound, float2& z_bound)
{
    x_bound.x=std::numeric_limits<float>::max();
    x_bound.y=std::numeric_limits<float>::min();
    y_bound.x=std::numeric_limits<float>::max();
    y_bound.y=std::numeric_limits<float>::min();
    z_bound.x=std::numeric_limits<float>::max();
    z_bound.y=std::numeric_limits<float>::min();

    for(float3 pt: pc)
    {
        x_bound.x=std::min(x_bound.x,pt.x);
        x_bound.y=std::max(x_bound.y,pt.x);

        y_bound.x=std::min(y_bound.x,pt.y);
        y_bound.y=std::max(y_bound.y,pt.y);

        z_bound.x=std::min(z_bound.x,pt.z);
        z_bound.y=std::max(z_bound.y,pt.z);
    }
}
