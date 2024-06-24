//		MPI code in C++.
//		See [Gropp/Lusk/Skjellum, "Using MPI", p.33/41 etc.]
//		and  /opt/mpich/include/mpi2c++/comm.h  for details

#include "binaryIO.h"
#include "geom.h"
#include "getmatrix.h"
#include "jacsolve.h"
#include "userset.h"
#include "vdop.h"

#include <cassert>
#include <chrono>           // timing
#include <cmath>
#include <iostream>
#include <string>
#include <omp.h>
#include <vector>
using namespace std;
using namespace std::chrono;  // timing

///  Generates sparse matrices (CRS) and right hand side for the Laplace on the given mesh
///  and stores them in binary files.
///
///  - meshname("square_100")   reads ASCii mesh file square_100.txt
///  - Right hand side is defined via function fNice(x,y) in userset.h
///  - the initial mesh is refined (nmesh-1) 2 times by standard.
///  - ./generateCRS nmesh           --> nmesh-1 refinement steps

int main(int argc, char **argv )
{
	const string meshname("square_100");
	//const string meshname("square_06");
    int nmesh = 3;
    if (argc > 1)  nmesh = atoi(argv[1]);

    Mesh const mesh_c(meshname+".txt");
    bool ba = mesh_c.checkObtuseAngles();
    if (ba) cout << "mesh corrected" << endl;
    //mesh_c.Debug(); cout << endl << "#############################\n";

    gMesh_Hierarchy ggm(mesh_c, nmesh);

    for (size_t ll = 0; ll < ggm.size(); ++ll)
    {
        const Mesh &mesh = ggm[ll];
		cout << "Matrix for mesh " << ll << " with " << mesh.Nnodes() << " nodes."<< endl;
        FEM_Matrix SK(mesh);
        vector<double> uv(SK.Nrows(), 0.0);    // temperature
        vector<double> fv(SK.Nrows(), 0.0);    // r.h.s.
        SK.CalculateLaplace(fv);
        //SK.CheckRowSum();
        SK.CheckMatrix();
        mesh.SetValues(uv, [](double x, double y) -> double
        {
            return x *x * std::sin(2.5 * M_PI * y);
        } );
        SK.ApplyDirichletBC(uv, fv);

        const string binname(meshname+"_"+to_string(ll)+"_mat"+".bin");
        SK.writeBinary(binname);
        const string rhsname(meshname+"_"+to_string(ll)+"_rhs"+".bin");
        write_binVector(rhsname,fv);
        //SK.Debug();
    }

    return 0;
}

