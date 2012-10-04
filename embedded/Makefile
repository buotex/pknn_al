MEX = mex
CC = /usr/bin/g++
MEXSUFFIX  = mexglx
all : rfpred.$(MEXSUFFIX) writematrix.$(MEXSUFFIX)
#all : testrf
#MATLABHOME = /cs/local/generic/lib/pkg/matlab-7.3
CFLAGS     = -fPIC  -O2 -DMATLAB_MEX_FILE  #-DMX_COMPAT_32 \
               
LDFLAGS    = -shared -rdynamic

LIBS      = -lm -lluajit
INCLUDE   = -I/usr/local/include/luajit-2.0  '-I$(MATLABHOME)/extern/include'
LINK 	    = -L/usr/local/lib
MEXFLAGS  = CC='$(CC)' LD='$(CC)' LDFLAGS='$(LDFLAGS)' -cxx -largeArrayDims
ARMADILLO_LINK = -larmadillo




.PHONY : all

.FORCE:

#impex.o: impex.cpp
#	$(CXX) $(CFLAGS) -shared -fPIC $(ARMADILLO_LINK) -c $^

rfpred.$(MEXSUFFIX): rfpred.o #impex.o
	  $(MEX) $(MEXFLAGS) $(LINK) $(LIBS) -output rfpred $^
rfpred.o: rfpred.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -c $^

#testrf: testrf.cpp
#	$(CC) -o testrf  $^


writematrix.$(MEXSUFFIX): writematrix.o 
	  $(MEX) $(MEXFLAGS) $(LINK) $(LIBS) -output writematrix $^
writematrix.o: writematrix.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -c $^





clean:
	rm *.o
	rm *.mexa64
