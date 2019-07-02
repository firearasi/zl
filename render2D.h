#ifndef RENDER_H
#define RENDER_H
#include "camera.h"
#include "aabb.h"
#include <vector>
#include "ray.h"
#include <curand_kernel.h>
//#define TPB 32
using namespace std;

extern curandState* devStates2D;
void setupSeeds2D(int tx);
__global__ void renderAll2DKernel(float *d_pixels,int nx,int ny,float3 *d_pc,int len,camera* d_cam,float radius,int *d_mutex,int ns,curandState* globalState);
#endif
