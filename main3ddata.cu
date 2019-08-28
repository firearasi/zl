#include "count3D.h"
#include "readfile.h"
#include "utility.h"

#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <vector>
#include <stdio.h>
#include "camera.h"
#include "render.h"
#include "cuda_check_error.h"

#define TX 32
#define TY 32

using namespace std;

extern int divUp(int a, int b);

int *density;

__global__ void countDensityKernel(int *density, int dx,int dy,int dz,float3* d_pc,int pc_len,int x1,int y1,int z1,int x2,int y2,int z2,int radius,int *d_mutex)
{

    const int pixel_index = blockIdx.x*blockDim.x+threadIdx.x;
    const int pc_index = blockIdx.y*blockDim.y+threadIdx.y;
    if(pixel_index>=dx*dy*dz || pc_index>=pc_len)
        return;
    int zi=pixel_index%dz;
    int yi=(pixel_index/dz)%dy;
    int xi=pixel_index/(dz*dy);
    float x=x1+(float)(x2-x1)/(float)dx*xi;
    float y=y1+(float)(y2-y1)/(float)dy*yi;
    float z=z1+(float)(z2-z1)/(float)dz*zi;

    //printf("pc:%d, pixel:(%f,%f,%f)\n",pc_index, x,y,z);
    float3 pc=d_pc[pc_index];
    float xdiff=pc.x-x;
    float ydiff=pc.y-y;
    float zdiff=pc.z-z;
    if(xdiff*xdiff+ydiff*ydiff+zdiff*zdiff<=radius*radius)
    {
        //printf("Hit!\n");
        bool leave=true;
        while(leave)
        {
            if (0 == (atomicCAS(&d_mutex[pixel_index],0,1)))
            {

                density[pixel_index] += 1;

                leave=false;
                atomicExch(&d_mutex[pixel_index], 0);
            }
        }
    }
}

int main3DData(int radius,int granularity)
{
    std::vector<float3> pc = read_yxz("yxz.txt");

    int x0=15;
    int y0=15;
    int z0=15;


    aabb box=point_cloud_bounds(pc);
    int m=(int)(box.max().x-box.min().x)/x0+1;
    int n=(int)(box.max().y-box.min().y)/y0+1;
    int p=(int)(box.max().z-box.min().z)/z0+1;
    box.print();


    float3* d_pc;
    int len=pc.size();

    CudaSafeCall(cudaMallocManaged(&d_pc,  len*sizeof(float3)));
    CudaSafeCall(cudaMemcpy(d_pc, &pc[0], len*sizeof(float3),cudaMemcpyHostToDevice));

    int x1,y1,z1,x2,y2,z2;
    x1=box.min().x;
    x2=box.max().x+1;
    y1=box.min().y;
    y2=box.max().y+1;
    z1=box.min().z;
    z2=box.max().z+1;

    int dx,dy,dz;
    dx=(x2-x1)/granularity;
    dy=(y2-y1)/granularity;
    dz=(z2-z1)/granularity;


    int total=dx*dy*dz;

    const dim3 blockSize(TX,TY);
    const dim3 gridSize(divUp(total,TX),divUp(len,TY));

    printf("Number of pixels: %d",total);

    CudaSafeCall(cudaMallocManaged(&density, total*sizeof(int)));


    int *d_mutex=0;
    CudaSafeCall(cudaMallocManaged((void**)&d_mutex, total*sizeof(int)));
    CudaSafeCall(cudaMemset(d_mutex,0,total*sizeof(int)));

    countDensityKernel<<<gridSize, blockSize>>>(density,dx,dy,dz,d_pc,len,x1,y1,z1,x2,y2,z2,radius,d_mutex);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());

    //写进文件
    char filename[50];
    sprintf(filename,"density_granularity_%d_radius_%d.csv",granularity,radius);
    ofstream file;
    file.open(filename);
    file<<"x,y,z,density"<<endl;
    for(int pixel_index=0;pixel_index<total;pixel_index++)
    {
        int zi=pixel_index%dz;
        int yi=(pixel_index/dz)%dy;
        int xi=pixel_index/(dz*dy);
        float x=x1+(float)(x2-x1)/(float)dx*xi;
        float y=y1+(float)(y2-y1)/(float)dy*yi;
        float z=z1+(float)(z2-z1)/(float)dz*zi;
        file<<x<<","<<y<<","<<z<<","<<density[pixel_index]<<endl;
    }

    file.close();

    CudaSafeCall(cudaFree(density));
    CudaSafeCall(cudaFree(d_pc));
    return 0;
}
