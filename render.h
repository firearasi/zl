#ifndef RENDER_H
#define RENDER_H
#include "camera.h"
#include "aabb.h"
#include <vector>
#include "ray.h"
#include <curand_kernel.h>
#define TPB 32
using namespace std;

void setupSeeds(int tx);
float render(int i, int j, int nx, int ny, camera& cam, aabb* cells, int m, int n, int p, int ns=1);
float render2(int i,int j,int nx,int ny,camera& cam, float3* d_pc, int len,float radius);
__global__ void renderAllKernel(float *d_pixels, int nx, int ny, float3 *d_pc, int len,
    float *d_max_density, camera *d_cam, float radius, int *d_mutex,int ns,curandState* globalState);
__global__ void maxKernel(float* d_max,float* d_array, int len);

extern curandState* devStates;
#endif
