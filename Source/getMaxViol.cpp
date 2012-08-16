#include "mex.h"
#include <math.h>
//Computes the point with most violated constraints
//i.e. point i for which (b_{y{i}}-p_{y{i}}(i))+sum_{c!=y{i}}(p_c(i)-b_c)


void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray* prhs[])
{
  const mxArray *d_mx=mexGetVariablePtr("caller","d_a");
  double *d=mxGetPr(d_mx);
  //  int m=mxGetM(d_mx);
  int mn=mxGetM(d_mx);
    
  const mxArray *Kv_mx=mexGetVariablePtr("caller","K0v");
  double *K0v=mxGetPr(Kv_mx);
  int n=mxGetM(Kv_mx);
  int m=(int)(mn/n);

  const mxArray *Kz_mx=mexGetVariablePtr("caller","K0z");
  double *K0z=mxGetPr(Kz_mx);

  const mxArray *r1_mx=mexGetVariablePtr("caller","row1");
  double *row1=mxGetPr(r1_mx);

  const mxArray *r2_mx=mexGetVariablePtr("caller","row2");
  double *row2=mxGetPr(r2_mx);

  const mxArray *slack_mx=mexGetVariablePtr("caller","slack");
  double *slack=mxGetPr(slack_mx);
  double *viol_all;

  const mxArray *y_mx=mexGetVariablePtr("caller","y");
  double *y=mxGetPr(y_mx);

  double *Kv=(double *)malloc(sizeof(double)*mn);

  for(int i=0;i<n;i++){
    for(int p=0;p<m;p++){
      Kv[i*m+p]=K0z[p*n+i]*row1[p]+K0v[p*n+i]*(row2[p]+1);
    }
  }
  viol_all=(double *)malloc(sizeof(double)*mn);
  double *Kv_sum=(double *)malloc(sizeof(double)*n);
  for(int i=0;i<n;i++){
    Kv_sum[i]=0;
    for(int p=0;p<m;p++){
      Kv_sum[i]+=Kv[i*m+p];
    }
  }
  for(int ic=0;ic<mn;ic++){
    int i=((int)(ic/m));
    int c=ic%m;
    viol_all[ic]=d[ic]*Kv_sum[i];
    viol_all[ic]=viol_all[ic]+((y[i]-1)==c?-1:1)*Kv[i*m+c];
  }
  for(int ic=0;ic<mn;ic++){
    viol_all[ic]=viol_all[ic]-slack[ic];
  }
  double *viol;
  plhs[0]=mxCreateDoubleMatrix(n,1,mxREAL);
  viol=mxGetPr(plhs[0]);
  //viol=(double *)malloc(sizeof(double)*n);
  for(int i=0;i<n;i++){
    viol[i]=0;
  }
  for(int i=0;i<mn;i++){
    viol[(int)(i/m)]+=viol_all[i]>0?viol_all[i]:0;
  }
  free(viol_all);
  free(Kv_sum);
  free(Kv);
  return;
}
