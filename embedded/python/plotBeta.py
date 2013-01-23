import numpy as np
import numpy as np
import pylab as pl
import scipy.special as ss
 
def beta(a, b, mew):
    e1 = ss.gamma(a + b)
    e2 = ss.gamma(a)
    e3 = ss.gamma(b)
    e4 = mew ** (a - 1)
    e5 = (1 - mew) ** (b - 1)
    return (e1/(e2*e3)) * e4 * e5
 
def plot_beta(a, b, ax):
    Ly = []
    Lx = []
    mews = np.mgrid[0:1:100j]
    for mew in mews:
        Lx.append(mew)
        Ly.append(beta(a, b, mew))
    ax.plot(Lx, Ly, label="a=%f, b=%f" %(a,b))
 
def main():
    #plot_beta(0.1, 0.1)
    
    line = pl.Line2D([0.5,0.5],[0,3])
    fig = pl.figure()
    ax = fig.add_subplot(111)
 
    ax.annotate('decision boundary', xy=(0.5,0.7),  xycoords='data',
            xytext=(0.8, 0.5), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left', verticalalignment='mid',
           )
    plot_beta(3, 3, ax)
    plot_beta(.5,.5, ax)    
    ax.add_line(line)
    #plot_beta(2, 3)
    #plot_beta(8, 4)
    pl.xlim(0.0, 1.0)
    pl.ylim(0.0, 3.0)
    pl.legend()
    pl.show()
 
if __name__ == "__main__":
    main()
