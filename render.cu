#include "render.h"
#include <limits.h>
#define TX 32
#define TY 32
#define TZ 32

int divUp(int a, int b){return (a+b-1)/b;}

__global__ void cumulatedDensityKernel(ray *r, aabb *cells, int m, int n, int p, float* d_density, int* mutex)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    const int j = blockIdx.y*blockDim.y+threadIdx.y;
    const int k = blockIdx.z*blockDim.z+threadIdx.z;
    const int index = i+j*m+k*m*n;
    if((i>=m) || (j>=n) || (k>=p)) return;

    if(cells[index].hit(*r,-FLT_MAX,FLT_MAX))
    {
        printf("Hit!\n");
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
    float u=float(i)/float(nx);
    float v=float(j)/float(ny);
    ray* d_r=0;
    cudaMalloc(&d_r, sizeof(ray));
    ray r=cam.get_ray(u,v);
    cudaMemcpy(d_r, &r,sizeof(ray), cudaMemcpyHostToDevice);

    const dim3 blockSize(TX,TY,TZ);
    const int bx = divUp(m, TX);
    const int by = divUp(n, TY);
    const int bz = divUp(p, TZ);
    const dim3 gridSize(bx,by,bz);
    float* d_density=0;
    cudaMallocManaged(&d_density,sizeof(float));
    cudaMemset(d_density,0,sizeof(float));
    int *d_mutex=0;
    cudaMallocManaged(&d_mutex,sizeof(int));
    cumulatedDensityKernel<<<gridSize,blockSize>>>(d_r, cells, m,n,p,d_density,d_mutex);
    //printf("Density at pixel %d,%d: %f\n",i,j,*d_density);
    float density=*d_density;
    cudaFree(d_r);
    cudaFree(d_density);
    cudaFree(d_mutex);
    return density;
}
