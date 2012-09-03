#include "mex.h"
#include "matrix.h"
//#include "lua.h"
//#include "lauxlib.h"
//#include "lualib.h"
//#include "luajit.h"
#include "lua.hpp"
#include <armadillo>
#include <inttypes.h>
#include <stdio.h>
#include "randomforest.h"
void stackDump (lua_State *L) {
  int i;
  int top = lua_gettop(L);
  for (i = 1; i <= top; i++) {  /* repeat for each level */
    int t = lua_type(L, i);
    switch (t) {

      case LUA_TSTRING:  /* strings */
        printf("`%s'", lua_tostring(L, i));
        break;

      case LUA_TBOOLEAN:  /* booleans */
        printf(lua_toboolean(L, i) ? "true" : "false");
        break;

      case LUA_TNUMBER:  /* numbers */
        printf("%g", lua_tonumber(L, i));
        break;

      default:  /* other values */
        printf("%s", lua_typename(L, t));
        break;

    }
    printf("  ");  /* put a separator */
  }
  printf("\n");  /* end the listing */
}
lua_State * getLuaState() {
  static lua_State *L = 0;
  if (L == 0) {
    printf("Setting up Lua\n");
    L = lua_open();
    luaL_openlibs(L);
    luaL_dofile(L,"foo.lua");
  }


  return L;
}
//extern "C" {
//bool writeMatrix(double * mat, int rows, int cols);
//}
bool writeMatrix(double * mat, int rows, int cols, const char * filename) {
  printf("Beep Beep Beep");

  //convert matrix to armadillo matrix
  arma::mat wrapper(mat, rows, cols);
  arma::diskio::save_arma_binary(wrapper, filename);
}


void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray* prhs[]) {

  if (nlhs != 1 || nrhs != 6) { 
	  mexErrMsgTxt("rfpred.cpp: wrong number of input or output arguments");
  }
  std::cout << "nlhs: " << nlhs << " nrhs: " << nrhs << std::endl;
  /*
     setbuf(stdout, NULL); //for debugging

  //matlab-part 
  char * buf[100];
  getcwd(buf);
  //end matlab part
  printf("%s \n", buf);
  */


  double * kernelData, * trnIndData, * queIndData, * labelData;
  int kernelN, trnN, queN, numRuns, labelN, numClasses;
  //Setup phase 
  printf("blub \n");

  kernelData = mxGetPr(prhs[0]);
  kernelN = mxGetM(prhs[0]);


  trnIndData = mxGetPr(prhs[1]);
  trnN   = mxGetM(prhs[1]) * mxGetN(prhs[1]); //in case it's transposed


  queIndData = mxGetPr(prhs[2]);
  queN   = mxGetM(prhs[2]) * mxGetN(prhs[2]); //in case it's transposed

  labelData = mxGetPr(prhs[3]);
  labelN = mxGetM(prhs[3]) * mxGetN(prhs[3]); //in case it's transposed

  writeMatrix(kernelData, kernelN, kernelN, "kernel");
  writeMatrix(labelData, labelN, 1, "labels");
  
  numClasses =*(mxGetPr(prhs[4]));

  numRuns =  *(mxGetPr(prhs[5]));

  MultiArrayView<2, double> wkernel = wrapArray(kernelData, kernelN, kernelN); 
  MultiArrayView<2, double> wlabels = wrapArray(labelData, labelN, 1);
  //RandomForest<> forest = 
  //createForests(wkernel, wlabels, trnIndData, trnN, queIndData, queN);
  plhs[0] = mxCreateNumericMatrix(numRuns, queN, mxUINT32_CLASS, mxREAL);
  uint32_t * iout;
  iout = (uint32_t *) mxGetData(plhs[0]);

  countVotes(
      wkernel,
      wlabels,
      numClasses,
      trnIndData,
      trnN,
      queIndData,
      queN,
      numRuns,
      iout
      );
}

/* Current Modus Operandi:
 * Matlab passes the Kernel matrix, along with two index-sets to the C-function
 * mexFunction.
 *
 * The mexFunction has a pointer to a statically created lua-session (only
 * create it once, so we have consistens global variables)
 *
 * the C-Part then creates the necessary space by allocating a matrix in Matlab,
 * and passing that piece of memory to the lua-function to fill it up with
 * values.
 *
 * TODO:
 * Which data does the lua part need? The kernel itself is pretty much the only
 * data necessary, isn't it? Perhaps the raw data? (which is available but hard
 * to deal with)
 * Or better the image-features themselves, which can be extracted by using the
 * vlfeat library, though the kernel data _should_ be enough by itself.
 *
 * How would that work? Every extra training data should (and will) add extra
 * dimensionality to whatever underlying classification algorithm is working.
 *
 * If working on a random forest: Retrain it? If we're just working with pkNN,
 * forget that point and just create a random forest in lua every time.
 * */


/*
   void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray* prhs[]) {

   if (nlhs != 1 || nrhs != 6) { 
   mexErrMsgTxt("rfpred.cpp: wrong number of input or output arguments");
   }
   std::cout << "nlhs: " << nlhs << " nrhs: " << nrhs << std::endl;

//  setbuf(stdout, NULL); //for debugging

//matlab-part 
//char * buf[100];
//getcwd(buf);
//end matlab part
//printf("%s \n", buf);


//lua_State * L = getLuaState();
lua_State * L = lua_open();
luaL_openlibs(L);
luaL_dofile(L,"foo.lua");

lua_getglobal(L, "wrapper");



double * kernelData, * trnIndData, * queIndData, * labelData;
double kernelN, trnN, queN, numAl, labelN, numClasses;
//Setup phase 
printf("blub \n");

kernelData = mxGetPr(prhs[0]);
lua_pushlightuserdata(L, (void *) kernelData);
kernelN = mxGetM(prhs[0]);
lua_pushnumber(L, kernelN);

trnIndData = mxGetPr(prhs[1]);
lua_pushlightuserdata(L, (void *) trnIndData);
trnN   = mxGetM(prhs[1]) * mxGetN(prhs[1]); //in case it's transposed
lua_pushnumber(L, trnN);

queIndData = mxGetPr(prhs[2]);
lua_pushlightuserdata(L, (void *) queIndData);
queN   = mxGetM(prhs[2]) * mxGetN(prhs[2]); //in case it's transposed
lua_pushnumber(L, queN);

labelData = mxGetPr(prhs[3]);
lua_pushlightuserdata(L, (void *) labelData);
labelN = mxGetM(prhs[3]) * mxGetN(prhs[3]); //in case it's transposed
lua_pushnumber(L, labelN);

writeMatrix(kernelData, kernelN, kernelN, "kernel");
writeMatrix(labelData, labelN, 1, "labels");

numClasses =*(mxGetPr(prhs[4]));
lua_pushnumber(L, numClasses);

numAl =  *(mxGetPr(prhs[5]));

lua_pushnumber(L, numAl);

plhs[0] = mxCreateNumericMatrix(numAl, 1, mxUINT64_CLASS, mxREAL);
uint64_t * iout;
iout = (uint64_t *) mxGetData(plhs[0]);

lua_pushlightuserdata(L, (void *) iout);

stackDump(L);
//second argument: number of elements on stack that should be used
//1 - function, 2 + 2  +2 + 2 - data, 1 - numAl
int error = lua_pcall(L,11,0, 0);

if (error) {
  printf("ERROR ERROR\n");
  printf("%s \n", lua_tostring(L, -1));
  lua_pop(L,1);
  lua_close(L);
  return; 
}
double test = (double)lua_tonumber(L, -1);
lua_pop(L,1); 
lua_close(L);
//lua_close(L);//TODO: Potential memory leak, but the c file is kept in memory
//as well, so whatever I guess.
}
*/
