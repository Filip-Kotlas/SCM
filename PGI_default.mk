# Basic Defintions for using PGI-compiler suite sequentially
# requires setting of COMPILER=PGI_
# OPTIRUN = optirun


CC	= pgcc
CXX     = pgc++
F77	= pgfortran
LINKER  = ${CXX}

# on mephisto:
#CXXFLAGS  += -I/share/apps/atlas/include
#LINKFLAGS += -L/share/apps/atlas/lib
#LINKFLAGS   += -lcblas -latlas

#LINKFLAGS   += -lblas
# Der <cblas.h> Header muss mit extern "C" versehen werden, damit g++ alles findet.

WARNINGS = -Minform=warn
# -Wall -pedantic -Wextra -Weffc++ -Woverloaded-virtual -W -Wfloat-equal -Wshadow  -Wredundant-decls
#           -pedantic -Wunreachable-code -Wextra -Winline
#  -Wunreachable-code

#PGI_PROFILING = -Minfo=ccff,loop,vect,opt,intensity,mp,accel
PGI_PROFILING = -Minfo=ccff,accel,ipa,loop,lre,mp,opt,par,unified,vect,intensity
# -Minfo
# -Mprof=time
# -Mprof=lines
#        take care with option      -Msafeptr
CXXFLAGS += -O3 -std=c++17  ${WARNINGS}
#CXXFLAGS += -O3 -std=c++11  -DNDEBUG ${PGI_PROFILING} ${WARNINGS}
#  -fastsse  -fargument-noalias ${WARNINGS}  -msse3 -vec-report=3

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
# #  Simple run time profiling of your code
# #  CXXFLAGS += -g -pg
# #  LINKFLAGS += -pg


# Profiling options PGI, see: pgcollect -help
# CPU_PROF = -allcache
CPU_PROF = -time
# GPU_PROF = -cuda=gmem,branch,cc13 -cudainit
#GPU_PROF = -cuda=branch:cc20
#
PROF_FILE = pgprof.out

cache: prof

prof: ${PROGRAM}
	${OPTIRUN} ${BINDIR}pgcollect $(CPU_PROF) ./$^
	${OPTIRUN} ${BINDIR}pgprof -exe ./$^  $(PROF_FILE) &

info:
	pgaccelinfo -v
