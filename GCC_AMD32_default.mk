# Basic Defintions for using GNU-compiler suite sequentially
# requires setting of COMPILER=GCC_

CC	= gcc
CXX     = g++
F77	= gfortran
LINKER  = ${CXX}

# on mephisto:
#CXXFLAGS  += -I/share/apps/atlas/include
#LINKFLAGS += -L/share/apps/atlas/lib
#LINKFLAGS   += -lcblas -latlas

#LINKFLAGS   += -lblas
# Der <cblas.h> Header muss mit extern "C" versehen werden, damit g++ alles findet.


#WARNINGS = -pedantic -pedantic-errors -Wall -Wextra -Werror -Wconversion -Weffc++ -Woverloaded-virtual  -Wfloat-equal -Wshadow
WARNINGS = -pedantic -Wall -Wextra -Wconversion -Weffc++ -Woverloaded-virtual  -Wfloat-equal -Wshadow \
           -Wredundant-decls -Winline -fmax-errors=1
#           -Wunreachable-code
#  -Wunreachable-code
CXXFLAGS += -ffast-math -O3 -march=native -std=c++17 ${WARNINGS}
#CXXFLAGS += -Ofast -funroll-all-loops -std=c++17 ${WARNINGS}
#-msse3
# -ftree-vectorizer-verbose=2  -DNDEBUG
# -ftree-vectorizer-verbose=5
# -ftree-vectorize -fdump-tree-vect-blocks=foo.dump  -fdump-tree-pre=stderr

# CFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp -fdump-tree-vect-details
# CFLAGS	= -ffast-math -O3 -funroll-loops -DNDEBUG -msse3 -fopenmp -ftree-vectorizer-verbose=2
# #CFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
# FFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
# LFLAGS  = -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
LINKFLAGS   += -O3

# BLAS, LAPACK
OPENBLAS_DIR = /opt/openblas_GCCseq
#OPENBLAS_DIR = /opt/openblas_GCC
OPENBLAS_LIBDIR = ${OPENBLAS_DIR}/lib
OPENBLAS_INCDIR = ${OPENBLAS_DIR}/include
CXXFLAGS += -I${OPENBLAS_INCDIR}
LINKFLAGS += -L${OPENBLAS_LIBDIR} -lopenblas

# interprocedural optimization
CXXFLAGS += -flto
LINKFLAGS += -flto

# profiling tools
#CXXFLAGS  += -pg
#LINKFLAGS += -pg

default: ${PROGRAM}

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean:
	@rm -f ${PROGRAM} ${OBJECTS}

clean_all:: clean
	-@rm -f *_ *~ *.bak *.log *.out *.tar *.orig
	-@rm -r html

run: clean ${PROGRAM}
#	time  ./${PROGRAM}
#	./${PROGRAM}
	( export LD_LIBRARY_PATH=${OPENBLAS_LIBDIR}:${LD_LIBRARY_PATH} ; ./${PROGRAM} )
#            or  'export LD_LIBRARY_PATH=/opt/openblas_gcc/lib:${LD_LIBRARY_PATH}'  in your ~/.bashrc 

# tar the current directory
MY_DIR = `basename ${PWD}`
tar:
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
	valgrind --tool=callgrind --simulate-cache=yes ./$^
#	kcachegrind callgrind.out.<pid> &
	kcachegrind `ls -1tr  callgrind.out.* |tail -1`

# Check for wrong memory accesses, memory leaks, ...
# use smaller data sets
# no "-pg"  in compile/link options
mem: ${PROGRAM}
	valgrind -v --leak-check=yes --tool=memcheck --undef-value-errors=yes --track-origins=yes --log-file=$^.addr.out --show-reachable=yes ./$^

#  Simple run time profiling of your code
#  CXXFLAGS += -g -pg
#  LINKFLAGS += -pg
prof: ${PROGRAM}
	./$^
	gprof -b ./$^ > gp.out
#	kprof -f gp.out -p gprof &

#Trace your heap:
#> heaptrack ./main.GCC_
#> heaptrack_gui heaptrack.main.GCC_.<pid>.gz
heap: ${PROGRAM}
	heaptrack ./$^ 11
	heaptrack_gui  `ls -1tr  heaptrack.$^.* |tail -1` &



########################################################################
#  get the detailed  status of all optimization flags
info:
	echo "detailed  status of all optimization flags"
	$(CXX) --version
	$(CXX) -Q $(CXXFLAGS) --help=optimizers
