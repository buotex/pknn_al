function [x flag1]=minProjGrad_newton_mx(x0, i, slack, gamma, eps, lbl_i, K0ii)
% Projects onto all the constraints associated with a single point in pknn
% optimization problem (See Eq 3, Jain and Kapoor CVPR'09)
global Ap;
global bp;
global cp;
global dd_a;
m=length(x0);
t=1;
x=x0;
olderr=1000;
old_diff_x=1000;
verr=1000;
verr1=1000;
alpha=0;
v=zeros(length(x0),1);
x_old=x0;
maxitr=30;
flag1=1;
while(verr>eps&&verr1>eps)
    if(t>maxitr)
        flag1=-1;
        break;
    end
    [dval H_d H_a H_b flag]=compHessian_Grad(x, i, slack, lbl_i, K0ii, gamma);    
    
    if(flag==-1)
        alpha=alpha/2;
        x=x_old+alpha*v;
        x(find(x>0))=0;
        t=t+1;
        continue;   
    end
    [f g v]=calcFreeVars(x, dval, H_d, H_a, H_b, lbl_i, dd_a);
    
    aap=Ap.*(f.^2);
    bbp=2*Ap.*f.*g+bp.*f;
    ccp=Ap.*(g.^2)+bp.*g+cp;
    ggamma=2*gamma/(v'*v);
    sslack=slack'*x-2*x'*v/gamma;
    alpha=myNewton_fast(0, aap, bbp, ccp, sslack, ggamma);
    x_old=x;
    x=x+alpha*v;
    x(find(x>0))=0;
    err=norm(dval,'fro');
    verr=abs(err-olderr);
    diff_x=norm(x-x_old);
    verr1=abs(diff_x-old_diff_x);
    old_diff_x=diff_x;
    olderr=err;
    t=t+1;
end
if(flag1==-1)
    options1 = optimset('GradObj','on', 'Display', 'off', 'LargeScale', 'on', 'Hessian', 'on');
    [x fval2 fexit2]=fmincon(@(x)compHessian_Grad_Func(x, i, slack, lbl_i, K0ii, gamma), zeros(m,1), [],[],[],[],[],[zeros(m,1)],[], options1);
    flag1=1;
end