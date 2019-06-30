#include "render.h"
#include <limits.h>
#define TX 32
#define TY 32
#define TZ 32

int divUp(int a, int b){return (a+b-1)/b;}

__global__ void cumulatedDensityKernel(float3 o, float3 d, aabb *cells,float* d_density, int* mutex)
{
    const int index = blockIdx.x*blockDim.x+threadIdx.x;

    //printf("cumulatedDensityKernel index=%d\n",index);

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


float render(int i, int j, int nx, int ny, camera& cam, aabb* cells, int m, int n, int p)
{
    //printf("render %d,%d\n",i,j);
    float u=float(i)/float(nx);
    float v=float(j)/float(ny);

    ray r=cam.get_ray(u,v);

   /* const dim3 blockSize(TX,TY,TZ);
    const int bx = divUp(m, TX);
    const int by = divUp(n, TY);
    const int bz = divUp(p, TZ);
    const dim3 gridSize(bx,by,bz);*/
    float* d_density=0;
    cudaMalloc(&d_density,sizeof(float));
    cudaMemset(d_density,0,sizeof(float));
    int *d_mutex=0;
    cudaMalloc(&d_mutex,sizeof(int));
    cudaMemset(d_mutex,0,sizeof(int));
    int blocks=divUp(m*n*p,TX);
    cumulatedDensityKernel<<<blocks,TX>>>(r.origin(),r.direction(), cells,d_density,d_mutex);

    //testKernel<<<blocks,TX>>>(m,n,p);
    float density;
    cudaMemcpy(&density, d_density,sizeof(float),cudaMemcpyDeviceToHost);
    //fprintf(stderr,"Density at pixel %d,%d: %f\n",i,j,density);
    cudaFree(d_density);
    cudaFree(d_mutex);
    return density;
}
