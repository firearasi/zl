#include "readfile.h"
#include "utility.h"
#include <iostream>
#include <vector>
int main()
{
    std::vector<float3> pc = read_yxz("yxz.txt");
    float2 x_bound,y_bound,z_bound;
    point_cloud_bounds(pc, x_bound,y_bound,z_bound);
    cout << "x bound: " << x_bound.x<<" " << x_bound.y<<endl;
    cout << "z bound: " << y_bound.x<<" " << y_bound.y<<endl;
    cout << "z bound: " << z_bound.x<<" " << z_bound.y<<endl;
    return 0;
}
