# Basic Defintions for using INTEL compiler suite sequentially
# requires setting of COMPILER=ONEAPI_

# https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
# requires
# source /opt/intel/oneapi/setvars.sh

BINDIR = /opt/intel/oneapi/compiler/latest/linux/bin/
MKL_ROOT = /opt/intel/oneapi/mkl/latest/
#export KMP_AFFINITY=verbose,compact

#----------------------------------
# ubuntu LINUX
#CUDA_DIR = /usr/local/cuda/targets/x86_64-linux
CUDA_DIR = /usr/local/cuda

# special for manjaro linux: make MANJARO=1
ifeq ($(MANJARO), 1)
# manjaro LINUX
#CUDA_DIR = /opt/cuda/targets/x86_64-linux
CUDA_DIR = /opt/cuda/
endif

CUDA_INC = ${CUDA_DIR}/include
CUDA_LIB = ${CUDA_DIR}/lib64
#----------------------------------

#CC	= ${BINDIR}icc
CXX     = ${BINDIR}dpcpp
#F77	= ${BINDIR}ifort
LINKER  = ${CXX}

WARNINGS = -pedantic -Wall -Weffc++ -Woverloaded-virtual -Wfloat-equal -Wshadow 
#-wd2015,2012
          #-Winline -Wunreachable-code  -Wredundant-decls
CXXFLAGS +=  -std=c++17 -O3  -fma -DNDEBUG ${WARNINGS}
#CXXFLAGS +=  -std=c++17 -O3 -march=core-avx2  -fma -ftz -fomit-frame-pointer -DNDEBUG ${WARNINGS} -mkl
# -fast       # fast inludes also -ipo !
#CXXFLAGS +=  -xCore-AVX2 -qopt-dynamic-align  -fargument-noalias-global  -fargument-noalias-ansi-alias -align
#CXXFLAGS +=  -tp=zen
# -qopt-subscript-in-range
# -vec-threshold0
# -xCORE-AVX2
# -axcode COMMON-AVX512 -axcode MIC-AVX512 -axcode CORE-AVX512 -axcode CORE-AVX2
# -ipo

# Reports: https://software.intel.com/en-us/articles/getting-the-most-out-of-your-intel-compiler-with-the-new-optimization-reports
#CXXFLAGS +=  -qopt-report=5 -qopt-report-phase=vec,par

#CXXFLAGS +=  -qopt-report=5 -qopt-report-phase=cg
# Redirect report from *.optrpt to stderr
#    -qopt-report-file=stderr
# Guided paralellization
#    -guide -parallel
#    -guide-opts=string  -guide-par[=n]  -guide-vec[=n]
#    -auto-p32 -simd

# interprocedural optimization
#CXXFLAGS  += -ipo
#LINKFLAGS += -ipo

# annotated Assembler file
ANNOTED = -fsource-asm -S 

# OpenMP
CXXFLAGS += -qopenmp
## -qopt-report-phase=openmp
## -diag-enable=sc-full  -diag-file=filename -diag-file-append[=filename]
LINKFLAGS += -qopenmp

# use MKL by INTEL
CXXFLAGS +=  -qmkl=parallel
# LINKFLAGS += -L${BINDIR}../composer_xe_2013.1.117/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
LINKFLAGS += -O3 -qmkl=parallel

# CUDA-libs
#CXXFLAGS  += -I${CUDA_INC} --cuda-gpu-arch=sm_75
CXXFLAGS  += -I${CUDA_INC}
# --cuda-gpu-arch=sm_60
LINKFLAGS += -L${CUDA_LIB} -lcudart -lcublas
# -lcudart_static -ldl -lrt -pthread


default:	${PROGRAM}

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean:
	rm -f ${PROGRAM} ${OBJECTS} *.optrpt

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
	echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
	amplxe-gui &

# GUI for Memory and Thread analyzer (race condition)
inspector: ${PROGRAM}
	echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
	#${BINDIR}../inspector_xe_2013/bin64/inspxe-gui &
	inspxe-gui &

advisor:
	echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
#	https://software.intel.com/en-us/articles/intel-advisor-2017-update-1-what-s-new
	export ADVIXE_EXPERIMENTAL=roofline
	advixe-gui &

vtune:
	vtune-gui ./${PROGRAM} &

icc-info:
	icpc -# main.cpp




