#ifndef RENDER_H
#define RENDER_H
#include "camera.h"
#include "aabb.h"

float render(int i, int j, int nx, int ny, camera& cam, aabb* cells, int m, int n, int p, int ns=1);


#endif
