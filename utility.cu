#include "utility.h"
#include <limits>
#include <algorithm>

// 找出点云的上下界

int divUp(int a, int b){return (a+b-1)/b;}

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

const float3 color0=make_float3(0,0,0);
const float3 color2=make_float3(0,0,1);
const float3 color4=make_float3(0,1,1);
const float3 color6=make_float3(0,1,0);
const float3 color8=make_float3(1,1,0);
const float3 color10=make_float3(1,0,0);


float3 heat_color(float value, float max_value)
{
    float pct=value/max_value;
    float t;
    if(pct<0.20)
    {
        t=(pct-0.0)/(0.20-0.0);
        return lerp(color0, color2,t);
    }
    if(pct<0.40)
    {
        t=(pct-0.20)/(0.40-0.20);
        return lerp(color2, color4,t);
    }
    if(pct<0.60)
    {
        t=(pct-0.40)/(0.60-0.40);
        return lerp(color4, color6,t);
    }
    if(pct<0.80)
    {
        t=(pct-0.60)/(0.80-0.60);
        return lerp(color6, color8,t);
    }
    if(pct<1.00)
    {
        t=(pct-0.80)/(1.00-0.80);
        return lerp(color8, color10,t);
    }
    else
        return color10;
}
