function x=multiDualMin(ap, bp, cp, slack, gamma)
%Solves the problem: 
%min_{x>=0} -\sum_p log(a_px^2+b_px+c_p) - gamma log (gamma-u*x)
options = optimset('GradObj','on', 'Display', 'off', 'Hessian', 'on', 'TolFun', 1e-15, 'TolX', 1e-15,'MaxIter',1e+5,'MaxFunEvals',1e+5);
[x fval flag]=fminunc(@(x)multiDualFunc(x, ap, bp, cp, slack, gamma),0,options);
if(flag<=0)
    keyboard
end

function [val dval H]=multiDualFunc(x, ap, bp, cp, slack, gamma)
Z=ap*x^2+bp*x+cp;
Z1=x^2/(2*gamma)-x*slack;
if(length(find(Z<1e-10))>0)
    val=1e+15;
    dval=1e+15;
    H=1e+15;
    return;
end
val=-sum(log(Z))+Z1;
dZ=2*ap*x+bp;
dval=-sum(dZ./Z)+x/gamma-slack;
H1=sum(((2*ap*x+bp).^2)./(Z.^2)-2*ap./Z);
H2=1/gamma;
H=H1+H2;