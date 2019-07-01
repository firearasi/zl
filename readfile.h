#ifndef READFILE_H
#define READFILE_H

#include <iostream>
#include <fstream>
#include  <string>
#include <vector>

using namespace std;
vector<float3> read_yxz(string filename);
__global__ void renderAllKernel(float *d_pixels,int nx,int ny,float3 *d_pc,int len,float *d_max_density,int *d_mutex);

#endif // READFILE_H
