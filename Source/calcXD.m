function xd=calcXD(x, dd_a, y)
xd=x+dd_a'*x;
xd(y)=xd(y)-2*x(y);