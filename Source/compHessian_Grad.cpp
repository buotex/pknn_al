#include "mex.h"
#include <math.h>
//#define INVOCATION_NAME startGdb
//#include "startgdb.h"

// The entry point searched for and called by Matlab.  See also:
// www.mathworks.com/access/helpdesk/help/techdoc/apiref/mexfunction_c.html


void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray* prhs[])
{
  const mxArray *det_mx=mexGetVariablePtr("global","DetVec");
  double *det=mxGetPr(det_mx);
  int m=mxGetM(det_mx);
  double *vK0v=mxGetPr(mexGetVariablePtr("global", "vK0v"));
  double *dd_a=mxGetPr(mexGetVariablePtr("global", "dd_a"));
  const mxArray *K0v_mx=mexGetVariablePtr("global", "K0v");
  int n=mxGetM(K0v_mx);
  double *K0v=mxGetPr(K0v_mx);
  double *K0z=mxGetPr(mexGetVariablePtr("global", "K0z"));
  double *zK0z=mxGetPr(mexGetVariablePtr("global", "zK0z"));
  double *vK0z=mxGetPr(mexGetVariablePtr("global", "vK0z"));

  
  double *x=mxGetPr(prhs[0]);
  int idx=(int)(*((double *)mxGetPr(prhs[1])))-1;
  double *slack=mxGetPr(prhs[2]);
  int y=(int)(*((double *)mxGetPr(prhs[3])))-1;
  double K0ii= (double)(*((double *)mxGetPr(prhs[4])));
  double gamma= (double)(*((double *)mxGetPr(prhs[5])));

  plhs[0]=mxCreateDoubleMatrix(m,1,mxREAL);
  plhs[1]=mxCreateDoubleMatrix(m,1,mxREAL);
  plhs[2]=mxCreateDoubleMatrix(m,1,mxREAL);
  plhs[3]=mxCreateDoubleMatrix(m,1,mxREAL);
  plhs[4]=mxCreateDoubleMatrix(1,1,mxREAL);

  double *dval=mxGetPr(plhs[0]);
  double *H_d=mxGetPr(plhs[1]);
  double *H_a=mxGetPr(plhs[2]);
  double *H_b=mxGetPr(plhs[3]);
  double *flag=mxGetPr(plhs[4]);
  
  double *xd=(double *)malloc(sizeof(double)*m);
  double dx=0;
  for(int i=0;i<m;i++){
    dx+=dd_a[i]*x[i];
  }
  for(int i=0;i<m;i++){
    xd[i]=x[i]+dx;
  }
  xd[y]=xd[y]-2*x[y];
  
  double igamma=1/(2*gamma);
  double u_sum=0;
  double lc_sum=0;
  
  for(int i=0;i<m;i++){
    double p1=1-vK0z[i]-K0v[i*n+idx]*xd[i]/2;
    double Z=p1*p1-vK0v[i]*(zK0z[i]+K0ii*xd[i]*xd[i]/4+K0z[i*n+idx]*xd[i]);
    if(Z<0){
      *flag=-1;
      return;
    }
    double iZ=1/Z;
    double pd1=p1*K0v[i*n+idx];
    double p2=vK0v[i]*(K0z[i*n+idx]+K0ii*xd[i]/2);
    double u=(pd1+p2)*iZ;
    u_sum+=u;
    dval[i]=u+x[i]*igamma-slack[i];
    double lc=u*u+det[i]*iZ;
    H_d[i]=lc+igamma;
    H_a[i]=dd_a[i];
    H_b[i]=lc;
    lc_sum+=lc;
    if(i==y){
      dval[y]=dval[y]-2*u;
      H_b[y]=H_b[y]-2*lc;
    }
  }
  for(int i=0;i<m;i++){
    dval[i]+=u_sum*dd_a[i];
    H_b[i]+=0.5*lc_sum*dd_a[i];
  }
  *flag=1;
  free(xd);
  return;
}
