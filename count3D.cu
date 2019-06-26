#include "count3D.h"
#include "utility.h"
#include <stdio.h>
#define TPB 32

__global__ void count3DKernel(float3 *pc, int len, float3 lower, float3 upper, int m, int n, int p,int* counts, int *mutex)
{


    const int t= blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=len)
        return;
    int i= (int)(pc[t].x-lower.x)/(upper.x-lower.x)*m;
    int j= (int)(pc[t].y-lower.y)/(upper.y-lower.y)*n;
    int k= (int)(pc[t].z-lower.z)/(upper.z-lower.z)*p;

    //printf("Thread %2d: point(%f,%f,%f) is in cell(%d,%d,%d)\n", t,pc[t].x,pc[t].y,pc[t].z,i,j,k);

    int cell_index=i+j*m+k*m*n;
    bool leave=true;
    //mutex
    while(leave)
    {
        if (0 == (atomicCAS(&mutex[cell_index],0,1)))
        {
            counts[cell_index]++;
            printf("counts[%d,%d,%d]=%d\n", i,j,k, counts[cell_index]);

            leave=false;
            atomicExch(&mutex[cell_index], 0);
        }
    }
}

void count3D(const std::vector<float3>pc, int m, int n,int p, int *counts)
{
    int len = pc.size();
    aabb box=point_cloud_bounds(pc);
    box.print();
    float3* d_pc;

    cudaMalloc(&d_pc, len*sizeof(float3));
    cudaMemcpy(d_pc, &pc[0], len*sizeof(float3),cudaMemcpyHostToDevice);


    int* d_counts;

    cudaMalloc(&d_counts, m*n*p*sizeof(int));

    int blocks=(len+TPB-1)/TPB;

    int* mutex;//all threads share on mutex.
    cudaMallocManaged((void**)&mutex, m*n*p*sizeof(int));
    cudaMemset(mutex,0,m*n*p*sizeof(int));

    count3DKernel<<<blocks, TPB>>>(d_pc, len, box.min(), box.max(), m,n,p,d_counts,mutex);

    cudaMemcpy(counts, d_counts, m*n*p*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_pc);
    cudaFree(d_counts);
    cudaFree(mutex);
}
