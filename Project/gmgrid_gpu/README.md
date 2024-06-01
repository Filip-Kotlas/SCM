# gmgrid: Geometric multigrid on GPU

### Status on 2024-May-07
 - Only CPU code for geometric multigrid is available
 - **make run**
 - or **main.NVCC_ levels**  with number of levels **levels** in [0,7].

### Data structures/implementation for GPU
 - use data structures from cg_2:Framework for preconditioned solvers on GPU and CPU
 - suggested: class vec from vdop_gpu.h
 - suggested: class CRS_Matrix_GPU from crsmatrix_gpu.h
 - use cuBLAS, cuSPARSE and more libraries whenever possible.
 
### currect code structure
 - *main.cpp*
 - *binaryIO.cpp*  : reads CRS matrix and vector from files
 - *vdop.cpp*      : some basic vector operations on CPU, also in *utils.h*
 - *geom.cpp*      : reads the coarse geometry, performes mesh handling and includes mesh hierarchy
 - *getmatrix.cpp* : compressed row storage matrix and its generation from a 2D mesh
 - *cuthill_mckee_ordering.cpp* : graph reordering to minimize the bandwidth
 - *jacsolve.cpp*  : Jacobi solver/smoother for a linear system of equations with CRS matrix; multigrid solver

