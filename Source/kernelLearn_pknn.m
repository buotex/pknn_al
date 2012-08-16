function [KK_row1 KK_row2 zc v]= kernelLearn_pknn(C, K0, y, params, zc0)
% Learns a kernel for each class using optimization problem of Eq 3, Jain and Kapoor CVPR-09
gamma=params.gamma;
thres=params.thres;
zc=zc0;
itr_cp=params.itr_cp;
verbosity=params.verbosity;
maxItr=params.maxItr;
diff_thres=params.diff_thres;
eps=params.eps;
global DetVec;
global vK0v;
global K0v;
global K0z;
global zK0z;
global vK0z;
global onesm;
global Ap;
global bp;
global cp;
global dd_a;

if(verbosity>=2)
    fprintf('Running pKNN classifier...\n');
end

[n,n] = size(K0);
m=length(unique(y));
numcons = size(C,1);
onesm=ones(m,1);

slack=zeros(numcons,1);
v=zeros(n, m);
for i=1:m
    v(find(y==i)',i)=1/length(find(y==i));
end

d_a=zeros(numcons, 1);
sim_idx=find(C(:,3)==1);
dissim_idx=find(C(:,3)==-1);

d_a(sim_idx, :)=C(sim_idx,4);
d_a(dissim_idx, :)=-C(dissim_idx,4);

i_idx_start=zeros(n,1);
for i=1:n
    i_idx_start(i)=find(C(:,1)==i, 1);
end


K0v=K0*v;
vK0v=sum(v.*K0v);
vK0v=vK0v';
K0z=K0*zc;
zK0z=sum(zc.*K0z);
zK0z=zK0z';
vK0z=sum(v.*K0z);
vK0z=vK0z';
itr=1;
slack_sum=zeros(n,1);
itr_cp=min(itr_cp,n);


Z1=(ones(m,1)-vK0z).^2-vK0v.*zK0z;
row1=vK0v'./Z1';
row2=(zK0z.*vK0v+(1-vK0z).*vK0z)'./Z1';
viol=getMaxViol();
[viol_sort sort_idx]=sort(viol, 'descend');
mxviol=viol_sort(1);   
mxviol_old=mxviol;
i_itrs=sort_idx(1:itr_cp);
count_itrs=1;
while(true)
    i=i_itrs(count_itrs);
    count_itrs=count_itrs+1;
    i_idx=i_idx_start(i):i_idx_start(i)+m-1;
    i_idx=i_idx';
    dd_a=d_a(i_idx, :);
    [Ap bp cp]=FindABC(i, K0(i,i));
    DetVec=.5*(K0(i,i)*vK0v-K0v(i,:)'.^2);
    x0=zeros(length(i_idx),1);
    
    alpha1=minProjGrad_newton_mx(x0, i, slack(i_idx), gamma, eps, y(i), K0(i,i));
    
    xd=calcXD(alpha1, dd_a, y(i));
    zc(i,:)=zc(i,:)+xd'/2;
    slack(i_idx,1)=slack(i_idx,1)-alpha1/(2*gamma);
    slack_sum(i)=sum(slack(i_idx,1));
    zK0z=zK0z+K0(i,i)*xd.^2/4+K0z(i,:)'.*xd;
    vK0z=vK0z+K0v(i,:)'.*xd/2;
    updateVZ(K0(:,i), xd/2);
    
    if(mod(itr, itr_cp)==0)
        Z1=(ones(m,1)-vK0z).^2-vK0v.*zK0z;
        row1=vK0v'./Z1';
        row2=(zK0z.*vK0v+(1-vK0z).*vK0z)'./Z1';
        viol=getMaxViol();
        [viol_sort sort_idx]=sort(viol, 'descend');
        i_itrs=sort_idx(1:itr_cp);        
        count_itrs=1;
    end
    
    if(mod(itr, 10*itr_cp)==0)
        Z1=(ones(m,1)-vK0z).^2-vK0v.*zK0z;
        row1=vK0v'./Z1';
        row2=(zK0z.*vK0v+(1-vK0z).*vK0z)'./Z1';
        viol=getMaxViol();
        viol_sort=sort(viol, 'descend');
        mxviol=viol_sort(1);        
        if(verbosity>=2)
            fprintf('%d %f %f\n',itr,mxviol,norm(mxviol_old-mxviol));
        end
        if(mxviol<thres||norm(mxviol-mxviol_old)<diff_thres||itr>maxItr)
            break;
        end
        mxviol_old=mxviol;
    end
    itr=itr+1;
end


KK=zeros(2,2,m);
for i=1:m
    Z1(i)=(1-vK0z(i))^2-vK0v(i)*zK0z(i);
    KK(:,:,i)=-(1/Z1(i))*[1-vK0z(i) vK0v(i);zK0z(i) 1-vK0z(i)];
end
row1=reshape(KK(1,1,:),m,1).*vK0v+reshape(KK(1,2,:),m,1).*vK0z;
KK_row1=row1';
row2=reshape(KK(2,1,:),m,1).*vK0v+reshape(KK(2,2,:),m,1).*vK0z;
KK_row2=row2';