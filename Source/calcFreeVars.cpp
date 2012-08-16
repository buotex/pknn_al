#include "mex.h"
#include <math.h>
//#define INVOCATION_NAME startGdb
//#include "startgdb.h"

// The entry point searched for and called by Matlab.  See also:
// www.mathworks.com/access/helpdesk/help/techdoc/apiref/mexfunction_c.html

void compXD(double *x, double *d, int y, double *xd, int m){
  double dx=0;
  for(int i=0;i<m;i++){
    dx+=d[i]*x[i];
  }
  for(int i=0;i<m;i++){
    xd[i]=x[i]+dx;
  }
  xd[y]=xd[y]-2*x[y];
}
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray* prhs[])
{
  const mxArray *x_mx;
  double *x, *dval, *Hd, *Ha, *Hb, *dda, *v, *f, *g;
  int lbl;
  x_mx=prhs[0];
  int m=mxGetM(x_mx);
  x=mxGetPr(x_mx);
  dval=mxGetPr(prhs[1]);
  Hd=mxGetPr(prhs[2]);
  Ha=mxGetPr(prhs[3]);
  Hb=mxGetPr(prhs[4]);
  dda=mxGetPr(prhs[6]);
  lbl=(int)(*((double *)mxGetPr(prhs[5]))-1);
  plhs[0]=mxCreateDoubleMatrix(m, 1, mxREAL);
  f=mxGetPr(plhs[0]);
  plhs[1]=mxCreateDoubleMatrix(m, 1, mxREAL);
  g=mxGetPr(plhs[1]);
  plhs[2]=mxCreateDoubleMatrix(m, 1, mxREAL);
  v=mxGetPr(plhs[2]);
  double iHdbd=0;
  double iHdad=0;
  double biHda=0;
  double aiHda=0;
  double biHdb=0;
  double temp1=0;
  double temp2=0;
  double *iHd=(double *)malloc(sizeof(double)*m);
  for(int i=0;i<m;i++){
    if(dval[i]<-1e-5&&fabs(x[i])<1e-5){
      ;
    }else{
      iHd[i]=1/Hd[i];
      temp1=Hb[i]*iHd[i];
      temp2=Ha[i]*iHd[i];
      iHdbd+=dval[i]*temp1;
      iHdad+=dval[i]*temp2;
      biHda+=Ha[i]*temp1;
      biHdb+=Hb[i]*temp1;
      aiHda+=Ha[i]*temp2;
    }
  }
  double biHda1=1+biHda;
  double det=pow(biHda1,2)-biHdb*aiHda;
  if(fabs(det)>1e-5){
    temp1=(biHda1*iHdbd-biHdb*iHdad)/det;
    temp2=(biHda1*iHdad-aiHda*iHdbd)/det;
  }else{
    double pinv_scale=(biHdb/pow(pow(biHdb,2)+pow(biHda1,2),2))*(biHda1*iHdbd+biHdb*iHdad);
    temp1=pinv_scale*biHdb;
    temp2=pinv_scale*biHda1;
  }
  
  for(int i=0;i<m;i++){
    if(dval[i]<-1e-5&&fabs(x[i])<1e-5){
      x[i]=0;
      v[i]=0;
    }else{
      v[i]=iHd[i]*(dval[i]-Ha[i]*temp1-Hb[i]*temp2);
      if(v[i]<0)
	v[i]=0;
    }
  }
  //  printf("%f %d %d %f %f\n", dval[lbl],lbl,m,v[lbl],fabs(x[lbl]));
  compXD(v, dda, lbl, f, m);
  compXD(x, dda, lbl, g, m);
}
