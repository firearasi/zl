#include "readfile.h"
#include "utility.h"
#include <iostream>
#include <vector>
#define TPB 32
using namespace std;

__global__ void countKernel(float3 *pc, float3 lower, float3 upper, int m, int n, int p)
{
    const int i= blockIdx.x*blockDim.x+threadIdx.x;
    printf("Thread %2d\n", i);
}

int main()
{
    std::vector<float3> pc = read_yxz("yxz.txt");
    aabb box=point_cloud_bounds(pc);
    box.print();

    float3* point_cloud_host = (float3 *)calloc(pc.size(), sizeof(float3));
    for(int i=0; i<pc.size();i++)
        point_cloud_host[i]=pc[i];
    float3* point_cloud;
    cudaMalloc(&point_cloud, pc.size()*sizeof(float3));
    cudaMemcpy(point_cloud, point_cloud_host, pc.size()*sizeof(float3),cudaMemcpyHostToDevice);

    int blocks=(pc.size()+TPB-1)/TPB;
    countKernel<<<blocks, TPB>>>(point_cloud, box.min(), box.max(), 30,30,30);

    cudaFree(point_cloud);
    free(point_cloud_host);

    return 0;
}
