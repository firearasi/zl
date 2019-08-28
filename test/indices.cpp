#include <iostream>
using namespace std;

int main()
{
	int lx,ly,lz;
	lx=5;ly=4;lz=3;
	int x,y,z;
	for(int i=0;i<lx*ly*lz;i++)
	{
		z=i%lz;
		y=(i/lz)%ly;
		x=i/(lz*ly);
		cout<<i<<": "<<x<<","<<y<<","<<z<<endl;

	}
	return 0;
}
