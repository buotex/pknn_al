from cffi import FFI
ffi = FFI()
ffi.cdef("""
         int createSingleDirich_(double * alpha, size_t length, size_t
         numSamples, double * result);
         """)
C = ffi.dlopen("../libuncertainty.so")


def getTUV(x, y, z):
    out = ffi.new("double[2]")
    alpha = ffi.new("double[3]")
    prior = 1./3
    alpha[0] = x + prior
    alpha[1] = y + prior
    alpha[2] = z + prior
    C.createSingleDirich_(alpha, 3, 1000, out)
    return out[0] - out[1]


import numpy as np

x,y,z = np.mgrid[0:10:100j, 0:10:100j, 0:10:100j]

vfunc = np.vectorize(getTUV)
retvol = vfunc(x,y,z)

from mayavi.mlab import *
obj = contour3d(x,y,z,retvol, contours = 10, opacity=.4)

import time

