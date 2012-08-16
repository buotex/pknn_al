#include "mex.h"
#include <math.h>
//#define INVOCATION_NAME startGdb
//#include "startgdb.h"

// The entry point searched for and called by Matlab.  See also:
// www.mathworks.com/access/helpdesk/help/techdoc/apiref/mexfunction_c.html

void multiDualFunc(double x, double *ap, double *bp, double *cp, int C, double slack, double gamma,double *diffval, double *val, double *dval){
  double *Z, *dZ;
  int i;
  Z=(double *)malloc(C*sizeof(double));
  dZ=(double *)malloc(C*sizeof(double));
  *val=x/gamma-slack;
  *dval=1/gamma;
  for(i=0;i<C;i++){
    Z[i]=ap[i]*x*x+bp[i]*x+cp[i];
    dZ[i]=2*ap[i]*x+bp[i];
    if(Z[i]<1e-10){
      *val=1e+15;
      *dval=1e+15;
      *diffval=1e+15;
      //      printf("%d %f here\n", i, x);
      return;
    }
    if(i==C-1){
      break;
    }
    *val=*val-dZ[i]/Z[i];
    *dval=*dval+dZ[i]*dZ[i]/(Z[i]*Z[i])-2*ap[i]/Z[i];
  }
  *diffval=fabs(*val-dZ[C-1]/Z[C-1]);
  free(Z);
  free(dZ);
}

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray* prhs[])
{
  const mxArray *ap_mx;
  const mxArray *bp_mx;
  const mxArray *cp_mx;
  double *ap;
  double *bp;
  double *cp;
  double *out;
  double gamma;
  double x0;
  double slack;
  int C,i,iter;
  double x;
  double diffval, val, dval;
  double Z1, dZ1;
  double u, v;
  double tol;
  ap_mx=prhs[1];
  bp_mx=prhs[2];
  cp_mx=prhs[3];
  ap=mxGetPr(ap_mx);
  bp=mxGetPr(bp_mx);
  cp=mxGetPr(cp_mx);
  C=mxGetM(ap_mx);
  gamma=(*mxGetPr(prhs[5]));
  x0=(*mxGetPr(prhs[0]));
  slack=(*mxGetPr(prhs[4]));
  x=x0;
  tol=1e-10;
  multiDualFunc(x, ap, bp, cp, C, slack, gamma, &diffval, &val, &dval);
  iter=0;
  plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
  out=mxGetPr(plhs[0]);
  while(diffval>tol){
    multiDualFunc(x, ap, bp, cp, C, slack, gamma, &diffval, &val, &dval);
    Z1=ap[C-1]*x*x+bp[C-1]*x+cp[C-1];
    dZ1=2*ap[C-1]*x+bp[C-1];
    if(diffval>=1e+12){
      x=-100;
      out[0]=x;
      return;
    }
    u=dval*Z1+val*dZ1;
    v=val*Z1-u*x;
    x=(v-bp[C-1])/(2*ap[C-1]-u);
    iter=iter+1;
  }
  out[0]=x;
  return;
}
