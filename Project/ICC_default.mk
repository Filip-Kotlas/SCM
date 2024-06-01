# Basic Defintions for using INTEL compiler suite sequentially
# requires setting of COMPILER=ICC_

#BINDIR = /opt/intel/bin/
CUDA_DIR = ${CUDA_PATH}
#/usr/local/cuda-10.2/targets/x86_64-linux
#CUDA_DIR = /opt/cuda/targets/x86_64-linux
CUDA_INC = ${CUDA_DIR}/include
CUDA_LIB = ${CUDA_DIR}/lib

# Error with g++-4.8 using icpc14.0,x:
#   find directory wherein bits/c++config.h is located
#   'locate bits/c++config.h'
#FOUND_CONFIG = -I/usr/include/x86_64-linux-gnu/c++/4.8


CC	= ${BINDIR}icc
CXX     = ${BINDIR}icpc
F77	= ${BINDIR}ifort
LINKER  = ${CXX}


WARNINGS = -Wall -Weffc++ -Woverloaded-virtual -Wfloat-equal -Wshadow -wd2015,2012 -wn3
#    -Winline -Wredundant-decls -Wunreachable-code
CXXFLAGS +=  -O3 -fargument-noalias -std=c++17 -DNDEBUG ${WARNINGS} -mkl ${FOUND_CONFIG}
# profiling tools
#CXXFLAGS  += -pg
#LINKFLAGS += -pg
# -vec-report=3
# -qopt-report=5 -qopt-report-phase=vec
# -guide -parallel
# -guide-opts=string  -guide-par[=n]  -guide-vec[=n]
# -auto-p32 -simd
CXXFLAGS += -align

# use MKL by INTEL
# LINKFLAGS += -L${BINDIR}../composer_xe_2013.1.117/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
LINKFLAGS += -O3 -L/opt/intel/mkl/lib -mkl

# OpenMP
CXXFLAGS += -fopenmp
LINKFLAGS += -fopenmp

# interprocedural optimization
CXXFLAGS += -ipo
LINKFLAGS += -ipo

# CUDA-libs
CXXFLAGS  += -I${CUDA_INC}
LINKFLAGS += -L${CUDA_LIB} -lcudart -lcublas

default:	${PROGRAM}

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean:
	@rm -f ${PROGRAM} ${OBJECTS}

clean_all:: clean
	@rm -f *_ *~ *.bak *.log *.out *.tar

run: clean ${PROGRAM}
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

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.f.o:
	$(F77) -c $(FFLAGS) -o $@ $<

##################################################################################################
# #    some tools
# # Cache behaviour (CXXFLAGS += -g  tracks down to source lines)
# cache: ${PROGRAM}
# 	valgrind --tool=callgrind --simulate-cache=yes ./$^
# #	kcachegrind callgrind.out.<pid> &
#
# # Check for wrong memory accesses, memory leaks, ...
# # use smaller data sets
# mem: ${PROGRAM}
# 	valgrind -v --leak-check=yes --tool=memcheck --undef-value-errors=yes --track-origins=yes --log-file=$^.addr.out --show-reachable=yes ./$^
#
# #  Simple run time profiling of your code
# #  CXXFLAGS += -g -pg
# #  LINKFLAGS += -pg
# prof: ${PROGRAM}
# 	./$^
# 	gprof -b ./$^ > gp.out
# #	kprof -f gp.out -p gprof &
#


mem: inspector
prof: amplifier
cache: amplifier

gap_par_report:
	${CXX}  -c -guide -parallel $(SOURCES) 2> gap.txt

# GUI for performance report
amplifier: ${PROGRAM}
	echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
#	alternatively to the solution above:
            #edit file  /etc/sysctl.d/10-ptrace.conf     and set variable   kernel.yama.ptrace_scope   variable to 0 .
	amplxe-gui &

# GUI for Memory and Thread analyzer (race condition)
inspector: ${PROGRAM}
	echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
	inspxe-gui &

advisor:
	echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
	advixe-gui &

icc-info:
	icpc -# main.cpp




