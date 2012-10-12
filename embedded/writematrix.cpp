#include <mex.h>
#include <armadillo>
bool writeMatrix(double * mat, int rows, int cols, const char * filename) {

  //convert matrix to armadillo matrix
  arma::mat wrapper(mat, rows, cols);
  arma::diskio::save_arma_binary(wrapper, filename);
}


void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray* prhs[]) {

  if (nrhs != 2) { 
    mexErrMsgTxt("writematrix: wrong number of input parameters");
  }

  char * inputBuf;

  double * data = mxGetPr(prhs[0]);
  int n = mxGetM(prhs[0]);
  int d = mxGetN(prhs[0]);
  if(mxIsChar(prhs[1])) {
    inputBuf = mxArrayToString(prhs[1]);
  }
  writeMatrix(data, n, d, inputBuf); 
}

