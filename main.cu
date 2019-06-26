#include "readfile.h"
#include "utility.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#define TPB 32
using namespace std;

__global__ void countKernel(float3 *pc, int len, float3 lower, float3 upper, int m, int n, int p,int* counts, int *mutex)
{
    const int t= blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=len)
        return;
    int i= (int)(pc[t].x-lower.x)/(upper.x-lower.x)*m;
    int j= (int)(pc[t].y-lower.y)/(upper.y-lower.y)*n;
    int k= (int)(pc[t].z-lower.z)/(upper.z-lower.z)*p;

    printf("Thread %2d: point(%f,%f,%f) is in cell(%d,%d,%d)\n", t,pc[t].x,pc[t].y,pc[t].z,i,j,k);
    //__syncthreads();
    int cell_index=i+j*m+k*m*n;
    bool leave=true;
    while(leave)
    {
        if (0 == (atomicCAS(&mutex[cell_index],0,1)))
        {
            counts[cell_index]++;
            printf("counts[%d]=%d\n", cell_index, counts[cell_index]);

            leave=false;
            atomicExch(&mutex[cell_index], 0);
        }
    }
}

int main()
{
    std::vector<float3> pc = read_yxz("yxz.txt");
    int len = pc.size();
    aabb box=point_cloud_bounds(pc);
    box.print();

    int m=60;
    int n=60;
    int p=60;
    float3* point_cloud_host = (float3 *)calloc(len, sizeof(float3));
    for(int i=0; i<len;i++)
        point_cloud_host[i]=pc[i];
    float3* d_pc;

    cudaMalloc(&d_pc, len*sizeof(float3));
    cudaMemcpy(d_pc, point_cloud_host, len*sizeof(float3),cudaMemcpyHostToDevice);

    int* counts;
    int* d_counts;
    counts=(int *)calloc(m*n*p,sizeof(int));
    cudaMalloc(&d_counts, m*n*p*sizeof(int));

    int blocks=(len+TPB-1)/TPB;

    int* mutex;//all threads share on mutex.
    cudaMallocManaged((void**)&mutex, m*n*p*sizeof(int));
    cudaMemset(mutex,0,m*n*p*sizeof(int));

    countKernel<<<blocks, TPB>>>(d_pc, len, box.min(), box.max(), m,n,p,d_counts,mutex);

    cudaMemcpy(counts, d_counts, m*n*p*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_pc);
    free(point_cloud_host);

    cudaFree(d_counts);

    FILE *file=fopen("result.csv","w");
    for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
            for(int k=0;k<p;k++)
    {
        fprintf(file,"%d,%d,%d,%d\n",i,j,k,counts[i]);
    }
    fclose(file);
    free(counts);
    cudaFree(mutex);

    return 0;
}
