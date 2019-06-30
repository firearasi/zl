//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#ifndef CAMERAH
#define CAMERAH
#include "ray.h"
#include <math.h>
#include "vec3.h"

class camera {
    public:
        camera(float3 lookfrom, float3 lookat, float3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
            lens_radius = aperture / 2;
            float theta = (float)vfov*M_PI/180;
            float half_height = tan(theta/2);
            float half_width = aspect * half_height;
            origin = lookfrom;
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);
            lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
            horizontal = 2*half_width*focus_dist*u;
            vertical = 2*half_height*focus_dist*v;
        }
        ray get_ray(float s, float t) {
            //float3 rd = lens_radius*random_in_unit_disk();
            //float3 offset = u * rd.x() + v * rd.y();
            float3 offset=make_float3(0,0,0);
            return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
        }

        float3 origin;
        float3 lower_left_corner;
        float3 horizontal;
        float3 vertical;
        float3 u, v, w;
        float lens_radius;
};
#endif
