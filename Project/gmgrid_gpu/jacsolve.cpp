//#include "vdop.h"
#include "vdop_gpu.h"
#include "geom.h"
#include "getmatrix.h"
#include "jacsolve.h"
#include "userset.h"
#include "crsmatrix_gpu.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;


void JacobiSolve_GPU(CRS_Matrix_GPU const &d_SK, Vec const &f, Vec &u)
{
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);
    double zero(0);
    double one(1);
    double minusone(-1);

    const double omega   = 1.0;
    //const int    maxiter = 1000;
    const int    maxiter = 240; // GH
    const double tol  = 1e-6;                // tolerance
    const double tol2 = tol * tol;           // tolerance^2

    int nrows = d_SK.Nrows();                  // number of rows == number of columns
    assert( nrows == static_cast<int>(f.size()) && f.size() == u.size() );

    cout << endl << " Start Jacobi solver for " << nrows << " d.o.f.s"  << endl;
    //  Choose initial guess
    u = vector<double>(u.size(), 0.0); 


    Vec dd(nrows);                // matrix diagonal
    Vec  r(nrows);                // residual
    Vec  w(nrows);                // correction

    d_SK.GetInvDiag(dd);                          //  dd := diag(K)

    //  Initial sweep
    d_SK.Defect(r, f, u);                      //  r := f - K*u
    vdmult_gpu(w, r, dd);                         //  w := D^{-1}*r

    double sigma0 = 0;
    cublasDdot( cublas_handle, nrows, w.data(), 1, r.data(), 1, &sigma0);      // s0 := <w,r>

    // Iteration sweeps
    int iter  = 0;
    double sigma = sigma0;
    while ( sigma > tol2 * sigma0 && maxiter > iter)  // relative error
    //while ( sigma > tol2 && maxiter > iter)         // absolute error
    {
        ++iter;
        cublasDaxpy(cublas_handle, nrows, &omega, w.data(), 1, u.data(), 1);            //  u := u + om*w
        d_SK.Defect(r, f, u);                  //  r := f - K*u
        vdmult_gpu(w, r, dd);                     //  w := D^{-1}*r
        double sig_old=sigma;
        cublasDdot( cublas_handle, nrows, w.data(), 1, r.data(), 1, &sigma);          // s0 := <w,r>
      	//cout << "Iteration " << iter << " : " << sqrt(sigma/sigma0) << endl;
        if (sigma>sig_old) cout << "Divergent at iter " << iter << endl;    // GH
    }

    cout << "aver. Jacobi rate :  " << exp(log(sqrt(sigma / sigma0)) / iter) << "  (" << iter << " iter)" << endl;
    cout << "final error: " << sqrt(sigma / sigma0) << " (rel)   " << sqrt(sigma) << " (abs)\n";

    return;
}

void JacobiSmoother_GPU(CRS_Matrix_GPU const &SK, Vec const &f, Vec &u,
                    Vec &r, int nsmooth, double const omega, bool zero)
{
    //// ToDO: ensure compatible dimensions
    SK.JacobiSmoother(f, u, r, nsmooth, omega, zero);
    return;
}

void DiagPrecond(CRS_Matrix_GPU const &SK, Vec const &r, Vec &w,
                 double const omega)
{
    // ToDO: ensure compatible dimensions
    Vec d_inv(r.size());
    SK.GetInvDiag(d_inv);        // accumulated diagonal of matrix @p SK.
    int const nnodes = static_cast<int>(w.size());  
    vdmult_elem(r.size(), w.data(), r.data(), d_inv.data() );
    cublasDscal(SK.get_cublasHandle(), r.size(), &omega, w.data(), 1);

    return;
}


Multigrid::Multigrid(Mesh const &cmesh, int const nlevel, vector<FEM_Matrix> matrices_for_setup,
                    vector<vector<double>> u, vector<vector<double>> f, vector<vector<double>> d, vector<vector<double>> w )
    : _meshes(cmesh, nlevel),
      _vSK(nlevel), _u(nlevel), _f(nlevel), _d(nlevel), _w(nlevel),
      _vPc2f()
{
    cout << "\n........................  in Multigrid::Multigrid  ..................\n";
    // Allocate Memory for matrices/vectors on all levels
    for (int lev = 0; lev < Nlevels(); ++lev) {
        matrices_for_setup.emplace_back( _meshes[lev] );  // CRS matrix
        const auto nn = matrices_for_setup.back().Nrows();
        u.resize(nn);
        f.resize(nn);
        d.resize(nn);
        w.resize(nn);
        auto vv = _meshes[lev].GetFathersOfVertices();
        cout << vv.size() << endl;
    }
    // Intergrid transfer operators
    _vPc2f.emplace_back( ); // no prolongation to the coarsest grid
    for (int lev = 1; lev < Nlevels(); ++lev) {
        _vPc2f.emplace_back( CRS_Matrix_GPU( BisectIntDirichlet( _meshes[lev].GetFathersOfVertices (), _meshes[lev-1].Index_DirichletNodes () ) ) );
        //checkInterpolation(lev);
        //checkRestriction(lev);
    }
    std::cout << "\n..........................................\n";
}

Multigrid::~Multigrid()
{}

void Multigrid::DefineOperators(vector<FEM_Matrix> matrices_for_setup, vector<vector<double>> u, vector<vector<double>> f )
{
    for (int lev = 0; lev < Nlevels(); ++lev) {
        DefineOperator(lev, matrices_for_setup[lev], u[lev], f[lev] );
    }
    return;
}

void Multigrid::DefineOperator(int lev, FEM_Matrix mat, vector<double> u, vector<double> f)
{
    //double tstart = omp_get_wtime(); 
    mat.CalculateLaplace( f);  // fNice()  in userset.h
    //double t1 = omp_get_wtime() - tstart;             // OpenMP
    //cout << "CalculateLaplace: timing in sec. : " << t1 << "   in level " << lev << endl;

    if (lev == Nlevels() - 1) {                // fine mesh
        _meshes[lev].SetValues( u, [](double x, double y) -> double
        { return x *x * std::sin(2.5 * M_PI * y); }
                              );
    }
    else {
        _meshes[lev].SetValues( u, f_zero);
    }    
    mat.ApplyDirichletBC( u, f);

    return;
}

void Multigrid::finish_setup(   vector<FEM_Matrix> matrices_for_setup,
                                vector<vector<double>> u,
                                vector<vector<double>> f,
                                vector<vector<double>> d, 
                                vector<vector<double>> w )
{
    int nlevels = matrices_for_setup.size();
    for(int i = 0; i < nlevels;i++)
    {
        std::cout << "finish setup: " << i << std::endl;
        _vSK.emplace_back( CRS_Matrix_GPU( matrices_for_setup[i] ) );
        _u.emplace_back( u[i] );
        _f.emplace_back( f[i] );
        _d.emplace_back( d[i] );
        _w.emplace_back( w[i] );
    }
}

void Multigrid::JacobiSolve(int lev)
{
    assert(lev < Nlevels());
    ::JacobiSolve_GPU(_vSK[lev], _f[lev], _u[lev]);
}

#include "timing.h"
void Multigrid::MG_Step(int lev, int const pre_smooth, bool const bzero, int nu)
{
    assert(lev < Nlevels());
    int const post_smooth = pre_smooth;

    if (lev == 0) { // coarse level
        // GH: a factorization (once in setup) with repeated forward-backward substitution would be better
        int n_jacobi_iterations = GetMesh(lev).Nnodes()/10; // ensure accuracy for coarse grid solver
        tic();
        JacobiSmoother_GPU(_vSK[lev], _f[lev], _u[lev], _d[lev],  n_jacobi_iterations, 1.0, true);
        cout << "     coarse grid solver [sec]: " << toc() << endl;
    }
    else {
        JacobiSmoother_GPU(_vSK[lev], _f[lev], _u[lev], _d[lev],  pre_smooth, 0.85, bzero || lev < Nlevels()-1);

        if (nu > 0) {
            _vSK[lev].Defect(_d[lev], _f[lev], _u[lev]);   //   d := f - K*u
            _vPc2f[lev].MultT(_d[lev], _f[lev - 1]);       // f_H := R*d
            // faster than Defect+MultT, slightly different final error (80 bit register for wi ?)
            //DefectRestrict(_vSK[lev], _vPc2f[lev], _f[lev - 1], _f[lev], _u[lev]); // f_H := R*(f - K*u)
            MG_Step(lev - 1, pre_smooth, true, nu);        // solve  K_H * u_H =f_H  with u_H:=0
            for (int k = 1; k < nu; ++k) {
                // W-cycle
                MG_Step(lev - 1, pre_smooth, false, nu);   // solve  K_H * u_H =f_H
            }
            _vPc2f[lev].Mult(_w[lev], _u[lev - 1]);         // w := P*u_H
            double one(1.0);
            cublasDaxpy(_vSK[lev].get_cublasHandle(), _u[lev].size(), &one, _w[lev].data(), 1, _u[lev].data(), 1 );  // u := u + tau*w
        }
        JacobiSmoother_GPU(_vSK[lev], _f[lev], _u[lev], _d[lev],  post_smooth, 0.85, false);
    }
    return;
}

void Multigrid::MG_Solve(int pre_smooth, double eps, int nu)
{
    int lev=Nlevels()-1;                // fine level

    double s0(-1);
    double si(0);
    // start with zero guess
    DiagPrecond(_vSK[lev], _f[lev], _u[lev], 1.0);  // w   := D^{-1]*f
    cublasDdot(_vSK[lev].get_cublasHandle(), _f[lev].size(), _f[lev].data(),1, _u[lev].data(), 1, &s0);     // s_0 := <f,u>

    bool bzero = true;                       // start with zero guess
    int  iter  = 0;
    do
    {
        MG_Step(lev, pre_smooth, bzero, nu);
        bzero=false;
        _vSK[lev].Defect(_d[lev], _f[lev], _u[lev]);    //   d := f - K*u
        DiagPrecond(_vSK[lev], _d[lev], _w[lev], 1.0);  // w   := D^{-1]*d
        cublasDdot(_vSK[lev].get_cublasHandle(), _d.size(), _d[lev].data(), 1, _w[lev].data(), 1, &si);       // s_i := <d,w>
        ++iter;
    } while (si>s0*eps*eps && iter<1000);

    cout << "\nrel. error: " << sqrt(si/s0) << "  ( " << iter << " iter.)" << endl;

    return;
}


// [[maybe_unused]] bool Multigrid::checkInterpolation(int const lev)
// {
//     assert(1<=lev && lev<Nlevels());
//     _meshes[lev-1].SetValues(_w[lev-1], [](double x, double y) -> double
//                            { return x+y; }  );*/
//     _meshes[lev].SetValues(_w[lev], [](double /*x*/, double /*y*/) -> double
//                            { return -123.0; }  );
//     //static_cast<BisectInterpolation>(_vPc2f[lev]).Mult(_d[lev], _w[lev - 1]);        // d := P*w_H
//     _vPc2f[lev].Mult(_d[lev], _w[lev - 1]);        // d := P*w_H
    
//     cout << "înterpolated  " << endl; GetMesh(lev).Visualize(_d[lev]);
    
//     return true;
// }

// [[maybe_unused]] bool Multigrid::checkRestriction(int const lev)
// {
//     assert(1<=lev && lev<Nlevels());
//     _meshes[lev].SetValues(_d[lev], [](double x, double y) 
//                            { return x+y; }  );
//     _meshes[lev-1].SetValues(_w[lev-1], [](double /*x*/, double /*y*/) -> double
//                            { return -123.0; }  );
//     //static_cast<BisectInterpolation>(_vPc2f[lev]).MultT(_d[lev], _w[lev - 1]);        // w_H := R*d
//     _vPc2f[lev].MultT(_d[lev], _w[lev - 1]);        // w_H := R*d
    
//     cout << "restricted  " << endl; GetMesh(lev-1).Visualize(_w[lev-1]);
    
//     return true;
// }
