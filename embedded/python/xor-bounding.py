import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib import colors, cm, interactive
size = 100
mean1= [10,10]
cov1 = [[1,0],[0,1]]
X1 = np.random.multivariate_normal(mean1, cov1, size)

sizel = 10
meanl= [10,10]
covl = [[4,0],[0,4]]
Xl = np.random.multivariate_normal(meanl, covl, sizel)

sizeu = 5
meanu= [0,0]
covu = [[1,0],[0,1]]
Xu = np.random.multivariate_normal(meanu, covu, sizeu)


sizeu2 = 5
meanu2= [20,20]
covu2 = [[1,0],[0,1]]
Xu2 = np.random.multivariate_normal(meanu2, covu2, sizeu2)

size2= 5
mean2= [0,0]
cov2 = [[1,0],[0,1]]
X2 = np.random.multivariate_normal(mean2, cov2, size2)


size2= 5
mean2= [20,20]
cov2 = [[1,0],[0,1]]
X3 = np.random.multivariate_normal(mean2, cov2, size2)

#mean2= [20,10]
#cov2 = [[1,0],[0,1]]
#X2 = np.random.multivariate_normal(mean2, cov2, size/10)
#Y2 = 2 * np.ones((X2.shape[0], 1))


fig = plt.figure()
ax = fig.add_subplot(111)

#Xfull = np.concatenate((X2,X1))
#Yfull = np.concatenate((Y2,Y1))
#random_queries = np.random.randint(0,Xfull.size, 50)

c1 = ax.scatter(X1[:,0], X1[:,1],
           marker= "x", color="red", alpha=0.3)
c2 = ax.scatter(X2[:,0], X2[:,1],
           marker= "x", color="blue", alpha=0.3)
c3 = ax.scatter(X3[:,0], X3[:,1],
           marker= "x", color="green", alpha=0.3)
l = ax.scatter(Xl[:,0], Xl[:,1],
           marker= "o", color="red")
u = ax.scatter(Xu[:,0], Xu[:,1],
           marker= "^", color="blue")
u = ax.scatter(Xu2[:,0], Xu2[:,1],
           marker= "^", color="green")
ax.legend((c1, c2, c3, l, u),("Class1, unlabeled", "Class2, unlabeled", "Class3, unlabeled", "Class1, labeled",
                              "random, Class0"), "upper left")

ax.annotate(
    "very uncertain, likely to be Class0", 
    xy = (0,0), xytext = (140, 20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

ax.annotate(
    "very uncertain, likely to be Class0", 
    xy = (20,20), xytext = (40, 20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

ax.annotate(
    "certain to be Class1", 
    xy = (10,10), xytext = (120, 20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

ax.annotate(
    "bounding box", 
    xy = (6,6), xytext = (-20, 20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

p = mpatches.Rectangle([6,6], 8,8, edgecolor="black", fill=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.add_patch(p)
fig.canvas.draw()
plt.show()
