#include "count3D.h"
#include "readfile.h"
#include "utility.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include "camera.h"
#include "render.h"
#include "cuda_check_error.h"

#define TX 32
#define TY 32

using namespace std;

extern int divUp(int a, int b);

int main3D(int ns)
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
/*
    int* counts;
    counts=(int *)calloc(m*n*p,sizeof(int));
    aabb* cells=0;
    CudaSafeCall(cudaMallocManaged((void**)&cells, m*n*p*sizeof(aabb)));
    count3D(pc, m, n,p, counts, cells);


 //write csv
    fprintf(stderr,"Writing cell distributions to cell_distribution.csv\n");
    FILE *file=fopen("cell_distribution.csv","w");
    fprintf(file,"i,j,k,lowerX,lowerY,lowerZ,upperX,upperY,upperZ,count\n");
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<p;k++)
    {
        float3 lower,upper;
        lower.x=box.min().x+(box.max().x-box.min().x)/m*i;
        upper.x=box.min().x+(box.max().x-box.min().x)/m*(i+1);
        lower.y=box.min().y+(box.max().y-box.min().y)/n*j;
        upper.y=box.min().y+(box.max().y-box.min().y)/n*(j+1);
        lower.z=box.min().z+(box.max().z-box.min().z)/p*k;
        upper.z=box.min().z+(box.max().z-box.min().z)/p*(k+1);
        cells[i+j*m+k*m*n]._min=lower;
        cells[i+j*m+k*m*n]._max=upper;
        fprintf(file,"%d,%d,%d,%f,%f,%f,%f,%f,%f,%d\n",
                i,j,k,
                lower.x,lower.y,lower.z,upper.x,upper.y,upper.z,
                (int)cells[i+j*m+k*m*n].density);
    }
    fclose(file);
    free(counts);
*/

 //volume rendering
    float3 centroid = make_float3(0.5*(box.min().x+box.max().x),
                                  0.5*(box.min().y+box.max().y),
                                  0.5*(box.min().z+box.max().z));
    //float3 origin = make_float3(-2200,1098,2210);
    //float3 origin=make_float3(500,2300,2210);
    float3 origin=make_float3(600,1100,4300);
    float3 unitY = make_float3(0,1,0);

    fprintf(stderr,"Centroid: (%f,%f,%f)\n",centroid.x,centroid.y,centroid.z);
    int nx=400;
    int ny=400;

    float radius =  40.0;

    setupSeeds(64);
    camera cam(origin,centroid,unitY,45,(float)nx/(float)ny,0,1000);
    //float max_density;
    //max_density=1.0f;

    ofstream pic;
    pic.open("pic.ppm");
    pic << "P3\n" << nx << " " << ny << "\n255\n";
    int ir,ig,ib;
    float *densities = (float *)calloc(nx*ny,sizeof(float));

    float3* d_pc;
    int len=pc.size();

    CudaSafeCall(cudaMallocManaged(&d_pc,  len*sizeof(float3)));
    CudaSafeCall(cudaMemcpy(d_pc, &pc[0], len*sizeof(float3),cudaMemcpyHostToDevice));

    float* d_pixels;
    CudaSafeCall(cudaMallocManaged(&d_pixels, nx*ny*sizeof(float)));
    CudaSafeCall(cudaMemset(d_pixels,0,nx*ny*sizeof(float)));

    float * d_max_density;
    CudaSafeCall(cudaMallocManaged(&d_max_density, sizeof(float)));
    CudaSafeCall(cudaMemset(d_max_density,0,sizeof(float)));



    const dim3 blockSize(TX,TY);
    const dim3 gridSize(divUp(nx*ny,TX),divUp(len,TY));

    int *d_mutex=0;
    CudaSafeCall(cudaMallocManaged((void**)&d_mutex, nx*ny*sizeof(int)));
    CudaSafeCall(cudaMemset(d_mutex,0,nx*ny*sizeof(int)));

    camera *d_cam;
    CudaSafeCall(cudaMallocManaged(&d_cam, sizeof(camera)));
    CudaSafeCall(cudaMemcpy(d_cam, &cam, sizeof(camera),cudaMemcpyHostToDevice));

    setupSeeds(TX);
    renderAllKernel<<<gridSize, blockSize>>>(d_pixels,nx,ny,d_pc,len,d_max_density,d_cam,radius,d_mutex,ns,devStates);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());
    printf("Max density before: %f\n",*d_max_density);


#if 1
    for(int j=ny-1;j>=0;j--)
        for(int i=0;i<nx;i++)
        {
            if(d_pixels[i+j*nx]>*d_max_density)
                *d_max_density=d_pixels[i+j*nx];
        }
    printf("Max density after: %f\n",*d_max_density);

#endif
    for(int j=ny-1;j>=0;j--)
        for(int i=0;i<nx;i++)
        {
            float3 color=heat_color(d_pixels[i+j*nx],*d_max_density);
            ir=int(255.99*color.x);
            ig=int(255.99*color.y);
            ib=int(255.99*color.z);
            pic << ir<<" " << ig<<" " << ib<<"\n";

        }
    free(densities);
    pic.close();
    //fprintf(stderr,"Max density: %f\n", max_density);

    //CudaSafeCall(cudaFree(cells));
    CudaSafeCall(cudaFree(d_pc));
    CudaSafeCall(cudaFree(d_pixels));
    CudaSafeCall(cudaFree(d_max_density));
    CudaSafeCall(cudaFree(d_mutex));
    CudaSafeCall(cudaFree(d_cam));

    return 0;
}
