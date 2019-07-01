#include "render.h"
#include <limits.h>
#include <stdlib.h>
//#include <curand_uniform.h>
#include <curand_kernel.h>
//#include <helper_cuda.h>
#include <time.h>
#define TX 64
#define TY 32
#define TZ 32

int divUp(int a, int b){return (a+b-1)/b;}

curandState* devStates=nullptr;
__global__ void setupSeedsKernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

void setupSeeds(int m, int n, int p)
{
    int blocks=divUp(m*n*p,TX);
    setupSeedsKernel<<<blocks,TX>>>(devStates,time(nullptr));

}

__global__ void cumulatedDensityKernel(float3 o, float3 d, aabb *cells,float* d_density, int* mutex, int ns,curandState* globalState)
{
    const int index = blockIdx.x*blockDim.x+threadIdx.x;

    curandState localState = globalState[index];
    float random1=curand_uniform(&localState)-0.5;
   // float random2=curand_uniform(NULL)-0.5;
    //float random3=curand_uniform(NULL)-0.5;

    if(cells[index].hit(o,d,0,FLT_MAX))
    {
    //    printf("Hit!\n");
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
    cudaMalloc(&d_density,sizeof(float));
    cudaMemset(d_density,0,sizeof(float));
    int *d_mutex=0;
    cudaMalloc(&d_mutex,sizeof(int));
    cudaMemset(d_mutex,0,sizeof(int));
    int blocks=divUp(m*n*p,TX);
    cumulatedDensityKernel<<<blocks,TX>>>(r.origin(),r.direction(), cells,d_density,d_mutex,ns,devStates);

    cudaMemcpy(&density, d_density,sizeof(float),cudaMemcpyDeviceToHost);
    //fprintf(stderr,"Density at pixel %d,%d: %f\n",i,j,density);
    cudaFree(d_density);
    cudaFree(d_mutex);

    return density;
}
