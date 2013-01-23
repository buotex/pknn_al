import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import colors, cm, interactive
from mpl_toolkits.mplot3d import Axes3D #<-- Note the capitalization! 

from pylab import *
def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

x = np.linspace(-2.0, 2)
y = np.linspace(-1, 1)
X, Y = np.meshgrid(x, y)
Z = gaussian_2d(X, Y, -1, 0, 0.9, 0.5)


x2 = np.linspace(0, 2.0)
y2 = np.linspace(0, 2.0)
X2, Y2 = np.meshgrid(x, y)
Z2 = gaussian_2d(X, Y, 1,0, 0.9, 0.5)

fig = plt.figure()
ax = fig.add_subplot(111)
CS = plt.contour(X, Y, Z, 2, colors='k')
CS2 = plt.contour(X2, Y2, Z2, 2, colors='k')
line = plt.Line2D([0,0],[-1,1], label="decbound")
ax.annotate('decision boundary', xy=(0,0.65),  xycoords='data',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
           )
ax.annotate(
    "candidate 1", 
    xy = (0,0), xytext = (-20, 20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

ax.annotate(
    "candidate 2", 
    xy = (0,-1), xytext = (-20, 20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
ax.add_line(line)
ax.scatter([0,0],[0,-1], marker="^")
fmt = {CS.levels[0]: '95%', CS.levels[1]: '99%'}
fmt2 = {CS2.levels[0]: '95%', CS2.levels[1]: '99%'}
plt.clabel(CS, fmt=fmt)
plt.clabel(CS2, fmt=fmt2)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
fig.savefig("gaussians.png")

