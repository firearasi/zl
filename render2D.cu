#include "render2D.h"
#include <limits.h>
#include <stdlib.h>
//#include <curand_uniform.h>

//#include <helper_cuda.h>
#include <time.h>
#include "cuda_check_error.h"
#include "utility.h"
#define TX 64
#define TY 32
#define TZ 32




curandState* devStates2D=nullptr;
__global__ void setupSeeds2DKernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

void setupPlaneSeeds(int tx)
{
    CudaSafeCall(cudaMalloc(&devStates2D, tx*sizeof(curandState)));
    setupSeeds2DKernel<<<1,tx>>>(devStates2D,time(nullptr));
    CudaCheckError();
}

__global__ void renderPlaneKernel(float *d_pixels,int nx,int ny,float3 *d_pc,int len, plane *d_plane,float* d_max_density,
                                  camera* d_cam, float radius,int *d_mutex,int ns,curandState* globalState)
{
    curandState localState = globalState[threadIdx.x];
    const int pixel_index = blockIdx.x*blockDim.x+threadIdx.x;
    const int pc_index = blockIdx.y*blockDim.y+threadIdx.y;
    if(pixel_index>=nx*ny || pc_index>=len)
        return;
    int i,j;

    i=pixel_index%nx;
    j=pixel_index/nx;

    for(int s=0;s<ns;s++)
    {
        float u,v;
        if(ns==1){
            u=float(i)/float(nx);
            v=float(j)/float(ny);
        }
        else
        {
            u=float(i+curand_uniform(&localState)-0.5)/float(nx);
            v=float(j+curand_uniform(&localState)-0.5)/float(ny);
        }
        ray r=d_cam->get_ray(u,v);
        float3 intersect_pt;
        if(!d_plane->intersects_with(r,intersect_pt))
            continue;

        if(length(intersect_pt-d_pc[pc_index])<=radius)
        {
            // printf("Hit!\n");

            bool go=true;
            while(go)
            {
                if (0 == (atomicCAS(&d_mutex[pixel_index],0,1)))
                {
                    d_pixels[pixel_index] += 1.0/ns;
                    *d_max_density = max(*d_max_density, d_pixels[pixel_index]);
                    atomicExch(&d_mutex[pixel_index], 0);

                    go=false;
                }
            }
        }
    }

}

