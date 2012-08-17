#include "mex.h"
#include "matrix.h"
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "luajit.h"
#include <inttypes.h>
#include <stdio.h>
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

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray* prhs[]) {

  if (nlhs != 1 || nrhs != 4) mexErrMsgTxt("rfpred.cpp: wrong number of input or output arguments");
  setbuf(stdout, NULL); //for debugging



  lua_State * L= lua_open();

  luaL_openlibs(L);
  char * buf[100];
  getcwd(buf);
  printf("%s \n", buf);
  luaL_dofile(L,"foo.lua");

  lua_getglobal(L, "wrapper");

  double * kernelData, * trnIndData, * queIndData;
  double kernelN, trnN, queN, numAl;
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
  numAl =  *(mxGetPr(prhs[3]));

  lua_pushnumber(L, numAl);
  
  plhs[0] = mxCreateNumericMatrix(numAl, 1, mxUINT64_CLASS, mxREAL);
  uint64_t * iout;
  iout = mxGetData(plhs[0]);
  
  lua_pushlightuserdata(L, (void *) iout);

  stackDump(L);
  
  int error = lua_pcall(L,8,0, 0);

  if (error) {
    printf("%s \n", lua_tostring(L, -1));
    lua_pop(L,1);
    lua_close(L);
    return; 
  }
  double test = (double)lua_tonumber(L, -1);
  lua_pop(L,1); 

  lua_close(L);

}
