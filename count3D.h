#ifndef COUNT3DKERNEL_H
#define COUNT3DKERNEL_H
#include <vector>
#include "aabb.h"

void count3D(const std::vector<float3>pc, int m, int n,int p, int *counts, aabb* cells);
__global__ void count3DKernel(float3 *pc, int len, float3 lower, float3 upper, int m, int n, int p,int* counts, aabb* cells,int *mutex);

#endif // COUNT3DKERNEL_H
