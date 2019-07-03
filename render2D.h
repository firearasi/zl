#ifndef RENDER2D_H
#define RENDER2D_H
#include "camera.h"
#include "aabb.h"
#include <vector>
#include "ray.h"
#include "plane.h"
#include <curand_kernel.h>
//#define TPB 32
using namespace std;

extern curandState* devStates2D;

void setupPlaneSeeds(int tx);
__global__ void renderPlaneKernel(float *d_pixels,int nx,int ny,float3 *d_pc,int len,plane *d_plane,float* d_max_density,
                                  camera* d_cam,float radius,int *d_mutex,int ns,curandState* globalState);
#endif
