function [A b c]=FindABC(i, K0ii)
global vK0v;
global K0v;
global K0z;
global zK0z;
global vK0z;
global onesm;

A=K0v(i,:)'.^2/4-vK0v*K0ii/4;
b=(vK0z-onesm).*K0v(i,:)'-vK0v.*K0z(i,:)';
c=(vK0z-onesm).^2-vK0v.*zK0z;