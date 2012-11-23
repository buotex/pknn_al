import math
import numpy 
import json
import zmq
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from mpl_toolkits.mplot3d import Axes3D #<-- Note the capitalization! 
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.metrics.pairwise import euclidean_distances
import graphdensity

numpy.set_printoptions(threshold='nan')

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
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


ctx = zmq.Context()
s = ctx.socket(zmq.REP)
s.bind("ipc:///tmp/zmq-test")
pref = s.recv()
s.send("OK")
X = recv_array(s)
Y = recv_array(s)
lda = LDA(n_components=X.shape[1])
lda.fit(X,Y)
XNew = lda.transform(X)
##
XDiff = numpy.sum(numpy.square(XNew[:,3:]), 1)
Y = XDiff
norm = None
cmap = cm.get_cmap(name="hot")
##
density = graphdensity.Density(X)
##

test = euclidean_distances(X, X)
test = numpy.reshape(test, (-1))
indices = numpy.argsort(test)
numItems = X.shape[0]
def convert(index, x, y):
    return (int(math.floor(index / x)), index % y)
#print(test[indices][numItems:numItems+100])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#print(pca.explained_variance_ratio_) 
#print(XNew)
i = 0
while True:
    i = i + 1

    trainingIndices= recv_array(s)
    trainingIndices = trainingIndices.reshape(-1).astype(numpy.int64)
    XTrained = XNew[trainingIndices, :]
    print(XTrained.shape)
    YTrained = Y[trainingIndices, :].squeeze()
    queryIndices = recv_array(s)
    queryIndices = queryIndices.reshape(-1).astype(numpy.int64)
    predictions = recv_array(s).squeeze()
    correctLabels = Y[queryIndices, :].squeeze()
    correctIndexLabels = numpy.where(correctLabels == predictions)
    incorrectIndexLabels = numpy.where(correctLabels != predictions)
    XQueried = XNew[queryIndices, :].squeeze()
    XCorrect = XQueried[correctIndexLabels[0],:].squeeze()
    YCorrect = correctLabels[correctIndexLabels[0]].squeeze()
    XInCorrect = XQueried[incorrectIndexLabels[0],:].squeeze()
    YInCorrect = correctLabels[incorrectIndexLabels[0]].squeeze()
    PredictionInCorrect= predictions[incorrectIndexLabels[0]].squeeze()

    pt0 = ax.scatter(XCorrect[:,0], XCorrect[:,1], XCorrect[:,2], 
                     c=YCorrect, 
                     marker=",",
                     cmap=cmap, norm = norm, alpha=0.4)
    pt1 = ax.scatter(XInCorrect[:,0], XInCorrect[:,1], XInCorrect[:,2],
                     c=YInCorrect,
                     marker="^",
                     cmap=cmap, norm = norm, alpha=0.7)
    pt2 = ax.scatter(XTrained[:,0], XTrained[:,1], XTrained[:,2],
                     c=YTrained,
                     marker="o",
                     s=40,
                     cmap=cmap, norm = norm)
    #fig.colorbar(sc)
    #canvas.print_figure(str.format("{}{:>03d}.png", pref,i), dpi=500)
    plt.show()
    fig.savefig(str.format("tmpresults/{}{:>03d}.png", pref,i), dpi=500)
    ax.set_autoscale_on(False)
    pt0.remove()
    pt1.remove()
    pt2.remove()

    pickedIndex = s.recv()
    density.pick(int(pickedIndex))
    send_array(s,density.getDensity())
    s.send("OK")
    #plt.clf()
