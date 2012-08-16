#include "mex.h"
#include <math.h>
//#define INVOCATION_NAME startGdb
//#include "startgdb.h"

// The entry point searched for and called by Matlab.  See also:
// www.mathworks.com/access/helpdesk/help/techdoc/apiref/mexfunction_c.html

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray* prhs[])
{
  const mxArray *Kz_mx=mexGetVariablePtr("caller","K0z");
  double *Kz, *Ki, *xd;
  int m=mxGetN(Kz_mx);
  int n=mxGetM(Kz_mx);
  Kz=mxGetPr(Kz_mx);
  Ki=mxGetPr(prhs[0]);
  xd=mxGetPr(prhs[1]);
  for(int j=0;j<m;j++){
    for(int i=0;i<n;i++){
      Kz[j*n+i]=Kz[j*n+i]+Ki[i]*xd[j];
    }
  }
//   for(int i=0;i<n;i++){
//     for(int j=0;j<m;j++){
//       Kz[i*m+j]=Kz[i*m+j]+Ki[i]*xd[j];
//     }
//   }
}
