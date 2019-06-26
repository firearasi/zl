#include "count3D.h"
#include "readfile.h"

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

    FILE *file=fopen("result.csv","w");
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<p;k++)
    {
        fprintf(file,"%d,%d,%d,%d\n",i,j,k,counts[i+j*m+k*m*n]);
    }
    fclose(file);
    free(counts);

    return 0;
}
