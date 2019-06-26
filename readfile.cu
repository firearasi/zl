#include "readfile.h"
vector<float3> read_yxz(string filename)
{
    string line;
    vector<float3> res;
    float3 pt;
    ifstream myfile (filename);
    if (myfile.is_open())
    {
         getline (myfile,line);
        while ( myfile)
        {

            myfile>>pt.y>>pt.x>>pt.z;
            res.push_back(pt);
            //cout << pt.x <<" "<<pt.y<<" "<<pt.z << '\n';
        }
        myfile.close();
    }
    return res;
}
