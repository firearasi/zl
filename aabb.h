#ifndef AABBH
#define AABBH

#include <iostream>
#include <vector>
float ffmin(float a, float b);
float ffmax(float a, float b);

class aabb{
public:
    aabb(){}
    aabb(const float3& a, const float3 &b)
	{	
        _min.x=ffmin(a.x,b.x);
        _min.y=ffmin(a.y,b.y);
        _min.z=ffmin(a.z,b.z);
        _max.x=ffmax(a.x,b.x);
        _max.y=ffmax(a.y,b.y);
        _max.z=ffmax(a.z,b.z);
	 	
	}
    float3 min() const {return _min;}
    float3 max() const {return _max;}

    static aabb surrounding_box(aabb box0, aabb box1)
	{
        float3 small, large;
        small.x=ffmin(box0.min().x, box1.min().x);
        small.y=ffmin(box0.min().y, box1.min().y);
        small.z=ffmin(box0.min().z, box1.min().z);

       large.x = ffmax(box0.max().x, box1.max().x);
       large.y = ffmax(box0.max().y, box1.max().y),
       large.z = ffmax(box0.max().z, box1.max().z);
	                                      				   
		return aabb(small, large);		   
	}
    bool contains(float3 pt);
	void print()
	{
        std::cerr << "("<<_min.x<<", "<<_min.y<<", "<<_min.z<<") - ";
        std::cerr << "("<<_max.x<<", "<<_max.y<<", "<<_max.z<<")\n";
	
	}
    float3 _min;
    float3 _max;
};

class aabb_mesh{
public:
    aabb_mesh(const float3& a, const float3& b,int _m, int _n, int _p)
        :box(a,b),m(_m),n(_n),p(_p){}
    int count_density_for_point_cloud(std::vector<float3>* pc, int i, int j, int k);
    aabb box;
    int m,n,p;
};


#endif
