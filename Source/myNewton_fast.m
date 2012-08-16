function [x fl]=myNewton_fast(x0, ap, bp, cp, slack, gamma)
%Solves the problem: 
% min_x -\sum_p log|ap x^2+bp x+cp|+x^2/(2*gamma)
% At each step: approximate function as -\sum_{p=1}^{c-1} (2 ap x+ bp)/(ap
% x^2+bp x+cp) + x/gamma = (u x+ v)/(ac x^2+bc x+cc)
x=myNewton_mex(x0, ap, bp, cp, slack, gamma);
fl=1;
if(x==-100)
    x=multiDualMin(ap, bp, cp, slack, gamma);
    fl=-1;
    return;
end
