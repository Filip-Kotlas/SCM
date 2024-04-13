# Basic Defintions for using INTEL compiler suite sequentially
# requires setting of COMPILER=ONEAPI_

#         https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
# requires
# source /opt/intel/oneapi/setvars.sh
# on  AMD:    export MKL_DEBUG_CPU_TYPE=5

#BINDIR = /opt/intel/oneapi/compiler/latest/linux/bin/
#MKL_ROOT = /opt/intel/oneapi/mkl/latest/
#export KMP_AFFINITY=verbose,compact

CC	= ${BINDIR}icc
CXX     = ${BINDIR}dpcpp
F77	= ${BINDIR}ifort
LINKER  = ${CXX}

## Compiler flags
WARNINGS = -Wall -Weffc++ -Woverloaded-virtual -Wfloat-equal -Wshadow -pedantic
WARNINGS += -Wpessimizing-move -Wredundant-move
#-wd2015,2012,2014 -wn3
#    -Winline -Wredundant-decls -Wunreachable-code
# -qopt-subscript-in-range
# -vec-threshold0

CXXFLAGS += -O3 -std=c++17  ${WARNINGS}
#CXXFLAGS += -DMKL_ILP64  -I"${MKLROOT}/include"
#CXXFLAGS += -DMKL_ILP32  -I"${MKLROOT}/include"
LINKFLAGS += -O3

# interprocedural optimization
CXXFLAGS  += -ipo
LINKFLAGS += -ipo
LINKFLAGS += -flto

# annotated Assembler file
ANNOTED = -fsource-asm -S 

#architecture
CPU  = -march=core-avx2
#CPU += -mtp=zen
# -xCORE-AVX2
# -axcode COMMON-AVX512 -axcode MIC-AVX512 -axcode CORE-AVX512 -axcode CORE-AVX2
CXXFLAGS  += ${CPU}
LINKFLAGS += ${CPU}

# use MKL by INTEL
# https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
# sequential MKL
#                              use the 32 bit interface (LP64) instead of 64 bit interface (ILP64)
CXXFLAGS +=  -qmkl=sequential  -UMKL_ILP64
LINKFLAGS += -O3 -qmkl=sequential -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
#LINKFLAGS += -O3 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread

# shared libs:  https://aur.archlinux.org/packages/intel-oneapi-compiler-static
#     install intel-oneapi-compiler-static   
# or 
LINKFLAGS += -shared-intel


OPENMP = -qopenmp
CXXFLAGS += ${OPENMP}
LINKFLAGS += ${OPENMP}


# profiling tools
#CXXFLAGS  += -pg
#LINKFLAGS += -pg
# -vec-report=3
# -qopt-report=5 -qopt-report-phase=vec -qopt-report-phase=openmp
# -guide -parallel
# -guide-opts=string  -guide-par[=n]  -guide-vec[=n]
# -auto-p32 -simd

# Reports: https://software.intel.com/en-us/articles/getting-the-most-out-of-your-intel-compiler-with-the-new-optimization-reports
#CXXFLAGS +=  -qopt-report=5 -qopt-report-phase=vec,par
#CXXFLAGS +=  -qopt-report=5 -qopt-report-phase=cg
# Redirect report from *.optrpt to stderr
#    -qopt-report-file=stderr
# Guided paralellization
#    -guide -parallel
#    -guide-opts=string  -guide-par[=n]  -guide-vec[=n]
#    -auto-p32 -simd

## run time checks
# https://www.intel.com/content/www/us/en/develop/documentation/fortran-compiler-oneapi-dev-guide-and-reference/top/compiler-reference/compiler-options/offload-openmp-and-parallel-processing-options/par-runtime-control-qpar-runtime-control.html


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
#    some tools
# Cache behaviour (CXXFLAGS += -g  tracks down to source lines)
# https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/microarchitecture-analysis-group/memory-access-analysis.html

mem: inspector
prof: vtune
cache: inspector

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
#	inspxe-gui &
	vtune-gui ./${PROGRAM} &

advisor:
	source /opt/intel/oneapi/advisor/2021.2.0/advixe-vars.sh
#	/opt/intel/oneapi/advisor/latest/bin64/advixe-gui &
	advisor --collect=survey ./${PROGRAM} 
#	advisor --collect=roofline ./${PROGRAM} 
	advisor --report=survey --project-dir=./ src:r=./ --format=csv --report-output=./out/survey.csv

vtune:
	echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
#	https://software.intel.com/en-us/articles/intel-advisor-2017-update-1-what-s-new
	export ADVIXE_EXPERIMENTAL=roofline
	vtune -collect hotspots ./${PROGRAM}
	vtune -report hotspots -r r000hs > vtune.out
#	vtune-gui ./${PROGRAM} &	

icc-info:
	icpc -# main.cpp

# MKL on AMD
# https://www.computerbase.de/2019-11/mkl-workaround-erhoeht-leistung-auf-amd-ryzen/
#
# https://sites.google.com/a/uci.edu/mingru-yang/programming/mkl-has-bad-performance-on-an-amd-cpu
# export MKL_DEBUG_CPU_TYPE=5
# export MKL_NUM_THRAEDS=1
# export MKL_DYNAMIC=false
#  on Intel compiler
# http://publicclu2.blogspot.com/2013/05/intel-complier-suite-reference-card.html
