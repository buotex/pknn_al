import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages


cardinality = [463, 5, 35, 44, 51, 163, 244, 429, 20, 30]
fullcount = sum(cardinality)
filelist =  ["boundingboxed2", "constant"]

X_data = range(1,151)
Y_data = [[] for i in range(len(filelist))]
Y_data2 = [[] for i in range(len(filelist))]
for i,filename in enumerate(filelist):
    lines = open("../results/set1/yeast_" + filename + ".dat").readlines()
    for l in lines[1:]:
        cols= l.split()
        values = [float(k) for k in cols[1:]]
        average = sum(values) / len(values)
        Y_data[i].append(average)


        detrate = cols[0].split('-')[1]
        Y_data2[i].append(detrate)


fig = plt.figure()
ax = fig.add_subplot(121)

ax2 = fig.add_subplot(122)
ax.set_xlabel("#Training Samples")
ax.set_ylabel("Averaged Recall")

ax2.set_xlabel("#Training Samples")
ax2.set_ylabel("Class Detection Rate")

markers = ( "^", "o", "+", "<","D", "v")
colors = ("red", "green", "magenta", "blue")
for dat, marker, label,color in zip(Y_data, markers, filelist, colors):
    ax.plot(X_data, dat,color=color,marker=marker, label = label, linestyle=":")
for dat, marker, label,color in zip(Y_data2, markers, filelist,colors):
    ax2.plot(X_data, dat,color=color,marker=marker, label = label, linestyle=":")
ax.legend(loc=2)
ax2.legend(loc=2)
plt.show()
pp = PdfPages("../results/set2/yeast_comparison_detection.pdf")
pp.savefig(fig)
pp.close()

