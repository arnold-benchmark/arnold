import numpy as np
import cython
from cython.parallel import parallel, prange

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_points(int height, int width, float[:,:] depth, float fx, float fy, float cx, float cy):
    cdef points_cam = np.zeros((height, width, 3), dtype=np.double)
    cdef double[:, :, :] result_view = points_cam
    cdef int v, u, k
    cdef float d
    with nogil, parallel():
        for v in prange(height, schedule='static'):
            for u in range(width):
                d = depth[v,u]
                
                # https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/reference_conventions.html
                # rotate around x axis 180
                # point[2] = -d *100.0
                # point[0] = -( u - cx ) * point[2] / fx
                # point[1] = (v - cy) * point[2] / fy
                
                
                result_view[v, u, 2] = -d *100.0
                result_view[v, u, 0] = -( u - cx ) * result_view[v, u, 2] / fx
                result_view[v, u, 1] = (v - cy) * result_view[v, u, 2]/ fy

    return points_cam

