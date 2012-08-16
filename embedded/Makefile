MEX = mex
CXX = /usr/bin/clang++
MEXSUFFIX  = mexglx
all : libmatlab.$(MEXSUFFIX)
#MATLABHOME = /cs/local/generic/lib/pkg/matlab-7.3
MEX        = mex
CFLAGS     = -fPIC -Wno-c++11-compat #-DMX_COMPAT_32 \
               -DMATLAB_MEX_FILE

LIBS      = -lm -lluajit
INCLUDE   = -I$(MATLABHOME)/extern/include -I/usr/local/include/luajit-2.0
LINK 	    = -L/usr/local/lib
MEXFLAGS  = -cxx CC='$(CXX)' CXX='$(CXX)' LD='$(CXX)'






.PHONY : all

.FORCE:

libmatlab.$(MEXSUFFIX): matlab.o
	  $(MEX) $(MEXFLAGS) $(LINK) $(LIBS) -output libmatlab $^
matlab.o: matlab.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -c $^
