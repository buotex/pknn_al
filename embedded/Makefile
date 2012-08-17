MEX = mex
CC = /usr/bin/clang
MEXSUFFIX  = mexglx
all : rfpred.$(MEXSUFFIX)
#MATLABHOME = /cs/local/generic/lib/pkg/matlab-7.3
CFLAGS     = -fPIC  -DMATLAB_MEX_FILE  #-DMX_COMPAT_32 \
               
LDFLAGS    = -shared -rdynamic

LIBS      = -lm -lluajit
INCLUDE   = -I/usr/local/include/luajit-2.0  '-I$(MATLABHOME)/extern/include'
LINK 	    = -L/usr/local/lib
MEXFLAGS  = CC='$(CC)' LD='$(CC)' LDFLAGS='$(LDFLAGS)' -largeArrayDims




.PHONY : all

.FORCE:

rfpred.$(MEXSUFFIX): rfpred.o
	  $(MEX) $(MEXFLAGS) $(LINK) $(LIBS) -output rfpred $^
rfpred.o: rfpred.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $^
