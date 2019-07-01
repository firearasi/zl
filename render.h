#ifndef RENDER_H
#define RENDER_H
#include "camera.h"
#include "aabb.h"
#include <vector>
#include "ray.h"
using namespace std;

void setupSeeds(int tx);
float render(int i, int j, int nx, int ny, camera& cam, aabb* cells, int m, int n, int p, int ns=1);
float render2(int i,int j,int nx,int ny,camera& cam, float3* d_pc, int len,float radius);

#endif
