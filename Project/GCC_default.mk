# Basic Defintions for using GNU-compiler suite sequentially
# requires setting of COMPILER=GCC_

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
CUDA_LIB = ${CUDA_DIR}/lib


CC	= gcc
CXX     = g++
F77	= gfortran
LINKER  = ${CXX}

#WARNINGS = -pedantic -pedantic-errors -Wall -Wextra -Werror -Wconversion -Weffc++ -Woverloaded-virtual  -Wfloat-equal -Wshadow 
WARNINGS = -pedantic -Wall -Wextra -Wconversion -Weffc++ -Woverloaded-virtual  -Wfloat-equal -Wshadow \
           -Wredundant-decls -Winline -fmax-errors=1
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

# OpenMP
CXXFLAGS += -fopenmp
LINKFLAGS += -fopenmp

# BLAS, LAPACK
LINKFLAGS   += -llapack -lblas

# special for manjaro linux: make MANJARO=1
#MANJARO="`grep Manjaro /etc/issue|cut -d " " -f 1| cut -d "o" -f 1`"
#MANJARO=`grep -c Manjaro /etc/issue`
ifeq ($(MANJARO), 1)
LINKFLAGS   += -lcblas
endif

# interprocedural optimization
CXXFLAGS  += -flto
LINKFLAGS += -flto

# OpenMP
#CXXFLAGS  += -fopenmp -foffload=nvptx-none
#LINKFLAGS += -fopenmp -foffload=nvptx-none

# CUDA-libs
CXXFLAGS  += -I${CUDA_INC}
LINKFLAGS += -L${CUDA_LIB} -lcudart -lcublas

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
	-@rm -rf html

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
	zip -r ${MY_DIR}.zip ${MY_DIR} *default.mk ;\
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
	# Graphical interface
	# valkyrie

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

codecheck: $(SOURCES)
	cppcheck --enable=all --inconclusive --std=c++17 --suppress=missingIncludeSystem $^

top:
	nvtop

########################################################################
#  get the detailed  status of all optimization flags
info:
	echo "detailed  status of all optimization flags"
	$(CXX) --version
	$(CXX) -Q $(CXXFLAGS) --help=optimizers
	inxi -C
