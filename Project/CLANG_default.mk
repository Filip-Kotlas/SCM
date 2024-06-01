# Basic Defintions for using GNU-compiler suite sequentially
# requires setting of COMPILER=CLANG_

# ubuntu LINUX
#CUDA_DIR = /usr/local/cuda/targets/x86_64-linux

ifeq ($(UBUNTU),1)
# on UBUNTU
CUDA_DIR= /usr/local/cuda
CUDABIN = ${CUDA_DIR}/bin/
else
# on manjaro
CUDABIN = 
CUDA_DIR= /opt/cuda
endif

CUDA_INC = ${CUDA_DIR}/include
CUDA_LIB = ${CUDA_DIR}/lib64

#CLANG_DIR = /home/haasegu/Downloads/LLVM/install/bin/
CC	    = ${CLANG_DIR}clang
CXX     = ${CLANG_DIR}clang++
#CXX     = /usr/bin/clang++-6.0
#F77	= gfortran
LINKER  = ${CXX}

#http://clang.llvm.org/docs/UsersManual.html#options-to-control-error-and-warning-messages
#WARNINGS = -Weverything -Wno-c++98-compat -ferror-limit=3
WARNINGS =  -Weverything -Wno-sign-conversion -Wno-c++98-compat -Wno-c++98-compat-bind-to-temporary-copy -Wno-shorten-64-to-32 -Wno-padded -Wno-\#pragma-messages
#-fsyntax-only -Wdocumentation -Wconversion -Wshadow -Wfloat-conversion -pedantic
# switch off some warnings from CUDA header files
WARNINGS += -Wno-reserved-id-macro -Wno-c++98-compat-pedantic -Wno-documentation-unknown-command -Wno-old-style-cast\
            -Wno-documentation-deprecated-sync -Wno-documentation -Wno-zero-as-null-pointer-constant

CXXFLAGS += -ffast-math -O3 -march=native  -std=c++17 -ferror-limit=1 ${WARNINGS}
# -ftrapv

# BLAS, LAPACK
LINKFLAGS   += -llapack -lblas

# special for manjaro linux: make UBUNTU=0
ifeq (~$(UBUNTU),1)
# on manjaro
LINKFLAGS   += -lcblas
endif

# interprocedural optimization
CXXFLAGS += -flto
LINKFLAGS += -flto

# OpenMP
CXXFLAGS  += -fopenmp
#-fopenmp-targets=nvptx64 -nocudalib
#-omptargets=nvptx64sm_35-nvidia-linux
LINKFLAGS += -fopenmp

# CUDA-libs
#CXXFLAGS  += -I${CUDA_INC} --cuda-gpu-arch=sm_75
#CXXFLAGS  += -I${CUDA_INC} --cuda-gpu-arch=sm_60
CXXFLAGS  += -I${CUDA_INC}
LINKFLAGS += -L${CUDA_LIB} -lcudart -lcublas
# -lcudart_static -ldl -lrt -pthread

#   very good check
# http://clang.llvm.org/extra/clang-tidy/
#TIDYFLAGS = -checks='modernize*'
#   good check, see:  http://llvm.org/docs/CodingStandards.html#include-style
TIDYFLAGS = -checks=llvm-*,-llvm-header-guard -header-filter=.* -enable-check-profile -extra-arg="-std=c++17" -extra-arg="-I${CUDA_INC}"
#TIDYFLAGS = -checks=llvm-*,readability-*,-llvm-header-guard  -header-filter=.* -export-fixes=fixes.txt
#
#TIDYFLAGS = -checks='readability-*'  -header-filter=.*
#   ???
#TIDYFLAGS = -checks='cert*'  -header-filter=.*
#   MPI checks ??
#TIDYFLAGS = -checks='mpi*'
#   ??
#TIDYFLAGS = -checks='performance*'   -header-filter=.*
#TIDYFLAGS = -checks='portability-*'  -header-filter=.*
#TIDYFLAGS = -checks='readability-*'  -header-filter=.*

default: ${PROGRAM}

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean:
	@rm -f ${PROGRAM} ${OBJECTS}

clean_all:: clean
	@rm -f *_ *~ *.bak *.log *.out *.tar

codecheck:tidy_check
tidy_check:
	clang-tidy ${SOURCES} ${TIDYFLAGS} -- ${SOURCES}
# see also http://clang-developers.42468.n3.nabble.com/Error-while-trying-to-load-a-compilation-database-td4049722.html


run: clean ${PROGRAM}
#	time  ./${PROGRAM}
	./${PROGRAM}

# tar the current directory
MY_DIR = `basename ${PWD}`
tar: clean_all
	@echo "Tar the directory: " ${MY_DIR}
	@cd .. ;\
	tar cf ${MY_DIR}.tar ${MY_DIR} *default.mk ;\
	cd ${MY_DIR}
# 	tar cf `basename ${PWD}`.tar *

zip: clean_all
	@echo "Zip the directory: " ${MY_DIR}
	@cd .. ;\
	zip ${MY_DIR}.zip ${MY_DIR}/* *default.mk ;\
	cd ${MY_DIR}

doc:
	doxygen Doxyfile

#########################################################################
.PRECIOUS: .cu .h
.SUFFIXES: .cu .h .o

.cu.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.f.o:
	$(F77) -c $(FFLAGS) -o $@ $<

##################################################################################################
#    some tools
# Cache behaviour (CXXFLAGS += -g  tracks down to source lines; no -pg in linkflags)
cache: ${PROGRAM}
	valgrind --tool=callgrind --simulate-cache=yes ./$^
#	kcachegrind callgrind.out.<pid> &
	kcachegrind `ls -1tr  callgrind.out.* |tail -1`

# Check for wrong memory accesses, memory leaks, ...
# use smaller data sets
mem: ${PROGRAM}
	valgrind -v --leak-check=yes --tool=memcheck --undef-value-errors=yes --track-origins=yes --log-file=$^.addr.out --show-reachable=yes ./$^

#  Simple run time profiling of your code
#  CXXFLAGS += -g -pg
#  LINKFLAGS += -pg
prof: ${PROGRAM}
	perf record ./$^
	perf report
#	gprof -b ./$^ > gp.out
#	kprof -f gp.out -p gprof &


##############################################
# CUDA with clang, see
# https://llvm.org/docs/CompileCudaWithLLVM.html
# example:
#   clang++ -std=c++17 first_unified.cu --cuda-gpu-arch=sm_75  -L/opt/cuda/lib64  -lcudart_static -ldl -lrt -pthread

# see also   https://gist.github.com/sonots/5abc0bccec2010ac69ff74788b265086
prof2: ${PROGRAM}
	$(CUDABIN)nvprof --print-gpu-trace ./$^ 2> prof2.txt

NSYS_OPTIONS = profile --trace=cublas,cuda  --sample=none --cuda-memory-usage=true --cudabacktrace=all --stats=true
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html
prof3: ${PROGRAM}
	$(CUDABIN)nsys $(NSYS_OPTIONS) ./$^ 
	$(CUDABIN)nsys-ui `ls -1tr  report*.qdrep|tail -1`  &
	
prof4: ${PROGRAM}
	$(CUDABIN)nsys-ui ./$^ 
	
top:
	nvtop
		

