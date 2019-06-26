#include "count3D.h"
#include "readfile.h"
#include "utility.h"

#include <iostream>
#include <vector>
#include <stdio.h>


using namespace std;



int main()
{
    std::vector<float3> pc = read_yxz("yxz.txt");

    int m=60;
    int n=60;
    int p=60;

    int* counts;
    counts=(int *)calloc(m*n*p,sizeof(int));
    count3D(pc, m, n,p, counts);

    printf("Writing cell distributions to cell_distribution.csv\n");
    aabb box=point_cloud_bounds(pc);
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

        fprintf(file,"%d,%d,%d,%f,%f,%f,%f,%f,%f,%d\n",
                i,j,k,
                lower.x,lower.y,lower.z,upper.x,upper.y,upper.z,
                counts[i+j*m+k*m*n]);
    }
    fclose(file);
    free(counts);

    return 0;
}
