import numpy as np
import zmq
import json

import matplotlib.pyplot as plt
from matplotlib import colors, cm, interactive
from mpl_toolkits.mplot3d import Axes3D #<-- Note the capitalization! 
interactive(True)

lines = open("/home/bxu/packages/misc_code/color_table.txt").readlines()
color_table = []
for l in lines:
    cols = l.split()
    color_table.append([ float(i) / 255 for i in cols[1:]])

cmap = colors.ListedColormap(color_table)
bounds = range(1,12)


norm = colors.BoundaryNorm(bounds, len(bounds) - 1)
def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    socket.send(A, flags, copy=copy, track=track)
    return socket.recv()

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    s.send("OK")
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])





ctx = zmq.Context()
s = ctx.socket(zmq.REP)
s.bind("ipc:///tmp/zmq-test")
#Dataset 1
size = 500
mean1= [10,20]
cov1 = [[1,0],[0,1]]
X1 = np.random.multivariate_normal(mean1, cov1, size)
Y1 = np.ones((X1.shape[0], 1))

mean2= [20,10]
cov2 = [[1,0],[0,1]]
X2 = np.random.multivariate_normal(mean2, cov2, size)
Y2 = np.ones((X2.shape[0], 1))

mean3= [10,10]
cov3 = [[1,0],[0,1]]
X3 = np.random.multivariate_normal(mean3, cov3, size)
Y3 = 3 * np.ones((X3.shape[0], 1))

mean4= [20,20]
cov4 = [[0.001,0],[0,0.001]]
X4 = np.random.multivariate_normal(mean4, cov4, size * 3)
Y4 = 2 * np.ones((X4.shape[0], 1))


Xfull = np.concatenate((X4,X2,X3,X1))
Yfull = np.concatenate((Y4,Y2,Y3,Y1))
#End Dataset 1

#Dataset 2

#size = 500
#mean1= [17,10]
#cov1 = [[1,0],[0,1]]
#X1 = np.random.multivariate_normal(mean1, cov1, size)
#Y1 = np.ones((X1.shape[0], 1))
#
#mean2= [13,10]
#cov2 = [[1,0],[0,1]]
#X2 = np.random.multivariate_normal(mean2, cov2, size)
#Y2 = 2 * np.ones((X2.shape[0], 1))
#
#mean3= [15,5]
#cov3 = [[15,0],[0,15]]
#X3 = np.random.multivariate_normal(mean3, cov3, size)
#Y3 = 3 * np.ones((X3.shape[0], 1))
#
#mean4= [9,10]
#cov4 = [[1,0],[0,1]]
#X4 = np.random.multivariate_normal(mean4, cov4, size)
#Y4 = 1 * np.ones((X4.shape[0], 1))
#
#mean5= [21,10]
#cov5 = [[1,0],[0,1]]
#X5 = np.random.multivariate_normal(mean5, cov5, size)
#Y5 = 2 * np.ones((X5.shape[0], 1))
#
##mean4= [20,20]
##cov4 = [[0.001,0],[0,0.001]]
##X4 = np.random.multivariate_normal(mean4, cov4, size)
##Y4 = 2 * np.ones((X4.shape[0], 1))
#
#
#Xfull = np.concatenate((X2,X3,X1, X4, X5))
#Yfull = np.concatenate((Y2,Y3,Y1, Y4, Y5))



#End Dataset 2




pref = s.recv()

ret = send_array(s, Xfull)
print(ret)
ret= send_array(s, Yfull)
print(ret)
s.send("OK")

fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111)
ax2 = fig2.add_subplot(111)
#ax.view_init(azim=-61, elev=52)
#ax.view_init(azim=-61, elev=52)

#print(pca.explained_variance_ratio_) 
#print(XNew)
i = 0

X = recv_array(s)
Y = recv_array(s)
XNew = X
while True:
    i = i + 1
    trainingIndices= recv_array(s)
    trainingIndices = trainingIndices.reshape(-1).astype(np.int64)
    XTrained = XNew[trainingIndices, :]
    print(XTrained.shape)
    YTrained = Y[trainingIndices, :].squeeze()
    queryIndices = recv_array(s)
    queryIndices = queryIndices.reshape(-1).astype(np.int64)
    predictions = recv_array(s).squeeze()
    correctLabels = Y[queryIndices, :].squeeze()
    correctIndexLabels = np.where(correctLabels == predictions)
    incorrectIndexLabels = np.where(correctLabels != predictions)
    XQueried = XNew[queryIndices, :].squeeze()
    XCorrect = XQueried[correctIndexLabels[0],:].squeeze()
    YCorrect = correctLabels[correctIndexLabels[0]].squeeze()
    XInCorrect = XQueried[incorrectIndexLabels[0],:].squeeze()
    YInCorrect = correctLabels[incorrectIndexLabels[0]].squeeze()
    PredictionInCorrect= predictions[incorrectIndexLabels[0]].squeeze()
    
    
    pt0 = ax.scatter(XCorrect[:,0], XCorrect[:,1], 
                     c=YCorrect, 
                     marker=",",
                     cmap=cmap, norm = norm, alpha=0.4)

    pt1 = ax.scatter(XInCorrect[:,0], XInCorrect[:,1], 
                     c=YInCorrect,
                     marker="^",
                     cmap=cmap, norm = norm, alpha=0.7)

    pt2 = ax.scatter(XTrained[:,0], XTrained[:,1], 
                     c=YTrained,
                     marker="o",
                     s=40,
                     cmap=cmap, norm = norm)

    #fig.colorbar(sc)
    #canvas.print_figure(str.format("{}{:>03d}.png", pref,i), dpi=500)
    ax.legend((pt0, pt1, pt2), ('Correct', 'Incorrect', 'TrainingSamples'))
    fig.savefig(str.format("{}{:>03d}.png", pref,i), dpi=500)
    ax.set_autoscale_on(False)
    
    tuvs = recv_array(s).squeeze()
    counts = recv_array(s).squeeze()
    
    def onpick(event):
        print(event.ind[0])
        print(tuvs[event.ind[0]])
        print(counts[event.ind[0],:])

    
    cid = fig2.canvas.mpl_connect('pick_event', onpick)
    pt3 = ax2.scatter(XQueried[:,0], XQueried[:,1],
                    c = tuvs,
                    s = 20,
                    marker="^",
                    cmap = cm.get_cmap(name="hot"), picker = 5)
    pt4 = ax2.scatter(XTrained[:,0], XTrained[:,1], 
                     c=YTrained,
                     marker="o",
                     s=40,
                     cmap=cmap, norm = norm)

    fig.canvas.draw()
    fig2.canvas.draw()
    raw_input()
    fig2.canvas.mpl_disconnect(cid)
    fig2.savefig(str.format("{}_tuv_{:>03d}.png", pref,i), dpi=500)
    pt0.remove()
    pt1.remove()
    pt2.remove()
    pt3.remove()
    pt4.remove()
    #plt.clf()


