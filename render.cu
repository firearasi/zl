#include "render.h"
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


curandState* devStates=nullptr;
__global__ void setupSeedsKernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id]);
}

__global__ void cumulatedDensityKernel(float3 o, float3 d, aabb *cells,int len,float* d_density, int* mutex, int ns,curandState* globalState)
{
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
    //curandState localState = globalState[threadIdx.x];
    if(index>=len)
        return;
    if(cells[index].hit(o,d,0,FLT_MAX))
    {
        //mutex
        bool leave=true;
        while(leave)
        {
            if (0 == (atomicCAS(mutex,0,1)))
            {
                *d_density += cells[index].density;

                leave=false;
                atomicExch(mutex, 0);
            }
        }
    }

}



float render(int i, int j, int nx, int ny, camera& cam, aabb* cells, int m, int n, int p, int ns)
{
    //printf("render %d,%d\n",i,j);
    float density;
    float u=float(i)/float(nx);
    float v=float(j)/float(ny);

    ray r=cam.get_ray(u,v);


    float* d_density=0;
    CudaSafeCall(cudaMalloc(&d_density,sizeof(float)));
    CudaSafeCall(cudaMemset(d_density,0,sizeof(float)));
    int *d_mutex=0;
    CudaSafeCall(cudaMalloc(&d_mutex,sizeof(int)));
    CudaSafeCall(cudaMemset(d_mutex,0,sizeof(int)));
    int blocks=divUp(m*n*p,TX);
    cumulatedDensityKernel<<<blocks,TX>>>(r.origin(),r.direction(), cells,m*n*p,d_density,d_mutex,ns,devStates);
    CudaCheckError();

    CudaSafeCall(cudaMemcpy(&density, d_density,sizeof(float),cudaMemcpyDeviceToHost));
    //fprintf(stderr,"Density at pixel %d,%d: %f\n",i,j,density);
    CudaSafeCall(cudaFree(d_density));
    CudaSafeCall(cudaFree(d_mutex));

    return density;
}


__global__ void cumulatedDensityKernel2(float3 o, float3 d, float3* d_pc,int len,float* d_density, float radius,int* mutex)
{
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=len)
        return;
    ray r(o,d);
    if(r.distance_to_pt(d_pc[index])<=radius)
    {
      //printf("Hit!\n");

        bool leave=true;
        while(leave)
        {
            if (0 == (atomicCAS(mutex,0,1)))
            {
                *d_density = *d_density+1;
                leave=false;
                atomicExch(mutex, 0);
            }
        }
    }
}


float render2(int i,int j,int nx,int ny,camera& cam, float3* d_pc, int len,float radius)
{
    float density;
    float u=float(i)/float(nx);
    float v=float(j)/float(ny);
    ray r=cam.get_ray(u,v);

    float* d_density=0;
    CudaSafeCall(cudaMalloc(&d_density,sizeof(float)));
    CudaSafeCall(cudaMemset(d_density,0,sizeof(float)));
    int *d_mutex=0;
    CudaSafeCall(cudaMalloc(&d_mutex,sizeof(int)));
    CudaSafeCall(cudaMemset(d_mutex,0,sizeof(int)));

    int blocks = divUp(len,TX);
    cumulatedDensityKernel2<<<blocks, TX>>>(r.origin(),r.direction(),d_pc,len,d_density,radius,d_mutex);
    CudaCheckError();

    CudaSafeCall(cudaMemcpy(&density, d_density,sizeof(float),cudaMemcpyDeviceToHost));
    //fprintf(stderr,"Density at pixel %d,%d: %f\n",i,j,density);
    CudaSafeCall(cudaFree(d_density));
    CudaSafeCall(cudaFree(d_mutex));


    return density;
}


void setupSeeds(int tx)
{
    CudaSafeCall(cudaMalloc(&devStates, tx*sizeof(curandState)));
    setupSeedsKernel<<<1,tx>>>(devStates,time(nullptr));
    CudaCheckError();
}

__global__ void renderAllKernel(float *d_pixels,int nx,int ny,float3 *d_pc,int len,float *d_max_density,camera* d_cam,float radius,int *d_mutex,int ns,curandState* globalState)
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
        if(r.distance_to_pt(d_pc[pc_index])<=radius)
        {
            // printf("Hit!\n");

            bool leave=true;
            while(leave)
            {
                if (0 == (atomicCAS(&d_mutex[pixel_index],0,1)))
                {
                    d_pixels[pixel_index] += 1.0/ns;
                    *d_max_density=max(*d_max_density, d_pixels[pixel_index]);
                    leave=false;
                    atomicExch(&d_mutex[pixel_index], 0);
                }
            }
        }
    }

}



__global__ void maxKernel(float* d_max,float* d_array, int len)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= len) return;
    const int s_idx = threadIdx.x;

    __shared__ float s_array[TPB];
    s_array[s_idx]=d_array[idx];
    __syncthreads();
    if(s_idx==0)
    {
        float blockMax=0;
        for(int i=0;i<TPB;i++)
        {
            if(s_array[i]>blockMax)
            {
                blockMax=s_array[i];
            }
        }
        if(blockMax > *d_max)
        {
            atomicAdd(d_max, blockMax-*d_max);
        }
    }
}
