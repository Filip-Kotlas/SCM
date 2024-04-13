# Basic Defintions for using GNU-compiler suite sequentially
# requires setting of COMPILER=CLANG_

#CLANGPATH=//usr/lib/llvm-10/bin/
CC     = ${CLANGPATH}clang
CXX    = ${CLANGPATH}clang++
#CXX   = ${CLANGPATH}clang++ -lomptarget  -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=/opt/pgi/linux86-64/2017/cuda/8.0
#F77   = gfortran
LINKER = ${CXX}

#http://clang.llvm.org/docs/UsersManual.html#options-to-control-error-and-warning-messages
WARNINGS += -Weverything -Wno-c++98-compat -Wno-sign-conversion -Wno-date-time -Wno-shorten-64-to-32 -Wno-padded -ferror-limit=1
WARNINGS += -Wdocumentation -Wconversion -Wshadow -Wfloat-conversion -pedantic
#-fsyntax-only -Wdocumentation -Wconversion -Wshadow -Wfloat-conversion -pedantic

CXXFLAGS += -O3 -std=c++17 -ferror-limit=1 ${WARNINGS}
# don't use -Ofast
# -ftrapv
LINKFLAGS += -O3

# different libraries in Ubuntu or manajaró
ifndef UBUNTU
UBUNTU=1
endif

# BLAS, LAPACK
LINKFLAGS += -llapack -lblas
# -lopenblas
ifeq ($(UBUNTU),1)
# ubuntu
else
# on  archlinux
LINKFLAGS += -lcblas
endif

# interprocedural optimization
CXXFLAGS  += -flto
LINKFLAGS += -flto

#   very good check
# http://clang.llvm.org/extra/clang-tidy/
#   good check, see:  http://llvm.org/docs/CodingStandards.html#include-style
SWITCH_OFF=,-readability-magic-numbers,-readability-redundant-control-flow,-readability-redundant-member-init
SWITCH_OFF+=,-readability-redundant-member-init,-readability-isolate-declaration
#READABILITY=,readability*${SWITCH_OFF}
#TIDYFLAGS = -checks=llvm-*,-llvm-header-guard -header-filter=.* -enable-check-profile -extra-arg="-std=c++17" -extra-arg="-fopenmp"
TIDYFLAGS = -checks=llvm-*,-llvm-header-guard${READABILITY} -header-filter=.* -enable-check-profile -extra-arg="-std=c++17" -extra-arg="-fopenmp"
#TIDYFLAGS += -checks='modernize*
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

codecheck: tidy_check
tidy_check:
	clang-tidy ${SOURCES} ${TIDYFLAGS} -- ${SOURCES}
# see also http://clang-developers.42468.n3.nabble.com/Error-while-trying-to-load-a-compilation-database-td4049722.html

run: clean ${PROGRAM}
#	time  ./${PROGRAM} ${PARAMS}
	./${PROGRAM} ${PARAMS}

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
# Cache behaviour (CXXFLAGS += -g  tracks down to source lines; no -pg in linkflags)
cache: ${PROGRAM}
	valgrind --tool=callgrind --simulate-cache=yes ./$^  ${PARAMS}
#	kcachegrind callgrind.out.<pid> &
	kcachegrind `ls -1tr  callgrind.out.* |tail -1`

# Check for wrong memory accesses, memory leaks, ...
# use smaller data sets
mem: ${PROGRAM}
	valgrind -v --leak-check=yes --tool=memcheck --undef-value-errors=yes --track-origins=yes --log-file=$^.addr.out --show-reachable=yes ./$^  ${PARAMS}

#  Simple run time profiling of your code
#  CXXFLAGS += -g -pg
#  LINKFLAGS += -pg
prof: ${PROGRAM}
	perf record ./$^  ${PARAMS}
	perf report
#	gprof -b ./$^ > gp.out
#	kprof -f gp.out -p gprof &

codecheck: tidy_check
