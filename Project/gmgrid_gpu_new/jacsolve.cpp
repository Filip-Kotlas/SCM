#include "vdop.h"
#include "geom.h"
#include "getmatrix.h"
#include "jacsolve.h"
#include "userset.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

// #####################################################################
void JacobiSolve(CRS_Matrix const &SK, vector<double> const &f, vector<double> &u)
{
    const double omega   = 1.0;
    //const int    maxiter = 1000;
    const int    maxiter = 240; // GH
    const double tol  = 1e-6;                // tolerance
    const double tol2 = tol * tol;           // tolerance^2

    int nrows = SK.Nrows();                  // number of rows == number of columns
    assert( nrows == static_cast<int>(f.size()) && f.size() == u.size() );

    cout << endl << " Start Jacobi solver for " << nrows << " d.o.f.s"  << endl;
    //  Choose initial guess
    for (int k = 0; k < nrows; ++k) {
        u[k] = 0.0;                          //  u := 0
    }

    vector<double> dd(nrows);                // matrix diagonal
    vector<double>  r(nrows);                // residual
    vector<double>  w(nrows);                // correction

    SK.GetDiag(dd);                          //  dd := diag(K)

    //  Initial sweep
    SK.Defect(r, f, u);                      //  r := f - K*u

    vddiv(w, r, dd);                         //  w := D^{-1}*r
    const double sigma0 = dscapr(w, r);      // s0 := <w,r>

    // Iteration sweeps
    int iter  = 0;
    double sigma = sigma0;
    while ( sigma > tol2 * sigma0 && maxiter > iter)  // relative error
    //while ( sigma > tol2 && maxiter > iter)         // absolute error
    {
        ++iter;
        vdaxpy(u, u, omega, w );             //  u := u + om*w
        SK.Defect(r, f, u);                  //  r := f - K*u
        vddiv(w, r, dd);                     //  w := D^{-1}*r
        double sig_old=sigma;
        sigma = dscapr(w, r);                // s0 := <w,r>
      	//cout << "Iteration " << iter << " : " << sqrt(sigma/sigma0) << endl;
        if (sigma>sig_old) cout << "Divergent at iter " << iter << endl;    // GH
    }
    cout << "aver. Jacobi rate :  " << exp(log(sqrt(sigma / sigma0)) / iter) << "  (" << iter << " iter)" << endl;
    cout << "final error: " << sqrt(sigma / sigma0) << " (rel)   " << sqrt(sigma) << " (abs)\n";

    return;
}

void JacobiSmoother(Matrix const &SK, std::vector<double> const &f, std::vector<double> &u,
                    std::vector<double> &r, int nsmooth, double const omega, bool zero)
{
    //// ToDO: ensure compatible dimensions
    SK.JacobiSmoother(f, u, r, nsmooth, omega, zero);
    return;
}

void DiagPrecond(Matrix const &SK, std::vector<double> const &r, std::vector<double> &w,
                 double const omega)
{
    // ToDO: ensure compatible dimensions
    auto const &D = SK.GetDiag();        // accumulated diagonal of matrix @p SK.
    int const nnodes = static_cast<int>(w.size());   
    for (int k = 0; k < nnodes; ++k) {
        w[k] = omega * r[k] / D[k];      // MPI: distributed to accumulated vector needed
    }

    return;
}


Multigrid::Multigrid(Mesh const &cmesh, int const nlevel)
    : _meshes(cmesh, nlevel),
      _vSK(), _u(_meshes.size()), _f(_meshes.size()), _d(_meshes.size()), _w(_meshes.size()),
      _vPc2f()
{
    cout << "\n........................  in Multigrid::Multigrid  ..................\n";
    // Allocate Memory for matrices/vectors on all levels
    for (int lev = 0; lev < Nlevels(); ++lev) {
        _vSK.emplace_back(_meshes[lev] );  // CRS matrix
        const auto nn = _vSK[lev].Nrows();
        _u[lev].resize(nn);
        _f[lev].resize(nn);
        _d[lev].resize(nn);
        _w[lev].resize(nn);
        auto vv = _meshes[lev].GetFathersOfVertices();
        cout << vv.size() << endl;
    }
    // Intergrid transfer operators
    _vPc2f.emplace_back( ); // no prolongation to the coarsest grid
    for (int lev = 1; lev < Nlevels(); ++lev) {
        _vPc2f.emplace_back( _meshes[lev].GetFathersOfVertices (), _meshes[lev-1].Index_DirichletNodes ()   );
        //checkInterpolation(lev);
        //checkRestriction(lev);
    }
    cout << "\n..........................................\n";
}

Multigrid::~Multigrid()
{}

void Multigrid::DefineOperators()
{
    for (int lev = 0; lev < Nlevels(); ++lev) {
        DefineOperator(lev);
    }
    return;
}

void Multigrid::DefineOperator(int lev)
{
    //double tstart = omp_get_wtime(); 
    _vSK[lev].CalculateLaplace(_f[lev]);  // fNice()  in userset.h
    //double t1 = omp_get_wtime() - tstart;             // OpenMP
    //cout << "CalculateLaplace: timing in sec. : " << t1 << "   in level " << lev << endl;

    if (lev == Nlevels() - 1) {                // fine mesh
        _meshes[lev].SetValues(_u[lev], [](double x, double y) -> double
        { return x *x * std::sin(2.5 * M_PI * y); }
                              );
    }
    else {
        _meshes[lev].SetValues(_u[lev], f_zero);
    }    
    _vSK[lev].ApplyDirichletBC(_u[lev], _f[lev]);

    return;
}

void Multigrid::JacobiSolve(int lev)
{
    assert(lev < Nlevels());
    ::JacobiSolve(_vSK[lev], _f[lev], _u[lev]);
}

#include "timing.h"
void Multigrid::MG_Step(int lev, int const pre_smooth, bool const bzero, int nu)
{
    assert(lev < Nlevels());
    int const post_smooth = pre_smooth;

    if (lev == 0) { // coarse level
        // GH: a factorization (once in setup) with repeated forward-backward substitution would be better
        int n_jacobi_iterations = GetMesh(lev).Nnodes()/10; // ensure accuracy for coarse grid solver
        //cout << "n_jacobi_iterations: " << n_jacobi_iterations << endl;
        //tic();
        JacobiSmoother(_vSK[lev], _f[lev], _u[lev], _d[lev],  n_jacobi_iterations, 1.0, true);
        //cout << "     coarse grid solver [sec]: " << toc() << endl;
    }
    else {
        JacobiSmoother(_vSK[lev], _f[lev], _u[lev], _d[lev],  pre_smooth, 0.85, bzero || lev < Nlevels()-1);
        if (nu > 0) {
            _vSK[lev].Defect(_d[lev], _f[lev], _u[lev]);   //   d := f - K*u
            _vPc2f[lev].MultT(_d[lev], _f[lev - 1]);       // f_H := R*d
            
        /*
        double sk_u, sk_d, sk_f, sk_w;
        int display_level = lev;
        dscapr(_u[display_level], _u[display_level], sk_u );
        dscapr(_d[display_level], _d[display_level], sk_d );
        dscapr(_f[display_level], _f[display_level], sk_f );        
        dscapr(_w[display_level], _w[display_level], sk_w );        
        cout << "sk_CPU: u: " << sk_u << ", d: " << sk_d << ", f: " << sk_f << ", w: " << sk_w << endl;
        */

            // faster than Defect+MultT, slightly different final error (80 bit register for wi ?)
            //DefectRestrict(_vSK[lev], _vPc2f[lev], _f[lev - 1], _f[lev], _u[lev]); // f_H := R*(f - K*u)
            MG_Step(lev - 1, pre_smooth, true, nu);        // solve  K_H * u_H =f_H  with u_H:=0
            for (int k = 1; k < nu; ++k) {
                // W-cycle
                MG_Step(lev - 1, pre_smooth, false, nu);   // solve  K_H * u_H =f_H
            }
            _vPc2f[lev].Mult(_w[lev], _u[lev - 1]);         // w := P*u_H
            vdaxpy(_u[lev], _u[lev], 1.0, _w[lev] );        // u := u + tau*w
        }
        JacobiSmoother(_vSK[lev], _f[lev], _u[lev], _d[lev],  post_smooth, 0.85, false);
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
    dscapr(_f[lev],_u[lev],s0);                     // s_0 := <f,u>

    bool bzero = true;                       // start with zero guess
    int  iter  = 0;
    do
    {
        MG_Step(lev, pre_smooth, bzero, nu);
        bzero=false;
        _vSK[lev].Defect(_d[lev], _f[lev], _u[lev]);    //   d := f - K*u
        DiagPrecond(_vSK[lev], _d[lev], _w[lev], 1.0);  // w   := D^{-1]*d
        dscapr(_d[lev],_w[lev], si);                // s_i := <d,w>
        ++iter;
        cout << "\nrel. error: " << sqrt(si/s0) << "  ( " << iter << " iter.)" << endl;
    } while (si>s0*eps*eps && iter<1000);

    cout << "\nrel. error: " << sqrt(si/s0) << "  ( " << iter << " iter.)" << endl;

    return;
}

[[maybe_unused]] bool Multigrid::checkInterpolation(int const lev)
{
    assert(1<=lev && lev<Nlevels());
    _meshes[lev-1].SetValues(_w[lev-1], [](double x, double y) -> double
                           { return x+y; }  );
    _meshes[lev].SetValues(_w[lev], [](double /*x*/, double /*y*/) -> double
                           { return -123.0; }  );
    //static_cast<BisectInterpolation>(_vPc2f[lev]).Mult(_d[lev], _w[lev - 1]);        // d := P*w_H
    _vPc2f[lev].Mult(_d[lev], _w[lev - 1]);        // d := P*w_H
    
    cout << "înterpolated  " << endl; GetMesh(lev).Visualize(_d[lev]);
    
    return true;
}

[[maybe_unused]] bool Multigrid::checkRestriction(int const lev)
{
    assert(1<=lev && lev<Nlevels());
    _meshes[lev].SetValues(_d[lev], [](double x, double y) 
                           { return x+y; }  );
    _meshes[lev-1].SetValues(_w[lev-1], [](double /*x*/, double /*y*/) -> double
                           { return -123.0; }  );
    //static_cast<BisectInterpolation>(_vPc2f[lev]).MultT(_d[lev], _w[lev - 1]);        // w_H := R*d
    _vPc2f[lev].MultT(_d[lev], _w[lev - 1]);        // w_H := R*d
    
    cout << "restricted  " << endl; GetMesh(lev-1).Visualize(_w[lev-1]);
    
    return true;
}


void JacobiSolve_GPU(CRS_Matrix_GPU const &SK, Vec const &f, Vec &u)
{
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    double zero(0);
    double one(1);
    double minusone(-1);

    const double omega   = 1.0;
    //const int    maxiter = 1000;
    const int    maxiter = 240; // GH
    const double tol  = 1e-6;                // tolerance
    const double tol2 = tol * tol;           // tolerance^2

    int nrows = SK.Nrows();                  // number of rows == number of columns
    assert( nrows == static_cast<int>(f.size()) && f.size() == u.size() );

    cout << endl << " Start Jacobi solver for " << nrows << " d.o.f.s"  << endl;
    //  Choose initial guess
    u = vector<double>(u.size(), 0.0); 


    Vec diag(nrows);                // matrix diagonal
    Vec  r(nrows);                // residual
    Vec  w(nrows);                // correction

    diag = SK.GetDiag();                          //  dd := diag(K)

    //  Initial sweep
    SK.Defect(r, f, u);                      //  r := f - K*u
    vddiv_gpu(w, r, diag);                         //  w := D^{-1}*r

    double sigma0 = 0;
    cublasDdot( SK.get_cublasHandle(), nrows, w.data(), 1, r.data(), 1, &sigma0);      // s0 := <w,r>

    // Iteration sweeps
    int iter  = 0;
    double sigma = sigma0;
    while ( sigma > tol2 * sigma0 && maxiter > iter)  // relative error
    //while ( sigma > tol2 && maxiter > iter)         // absolute error
    {
        ++iter;
        cublasDaxpy( SK.get_cublasHandle(), nrows, &omega, w.data(), 1, u.data(), 1);            //  u := u + om*w
        SK.Defect(r, f, u);                  //  r := f - K*u
        vddiv_gpu(w, r, diag);                     //  w := D^{-1}*r
        double sig_old=sigma;
        cublasDdot( SK.get_cublasHandle(), nrows, w.data(), 1, r.data(), 1, &sigma);          // s0 := <w,r>
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

void DiagPrecond_GPU(CRS_Matrix_GPU const &SK, Vec const &r, Vec &w, double const omega)
{
    assert( r.size() == w.size() );

    int const nnodes = static_cast<int>(w.size());  
    Vec diag(nnodes);
    diag = SK.GetDiag();        // accumulated diagonal of matrix @p SK.
    vddiv_gpu(w, r, diag );
    CHECK_CUBLAS( cublasDscal( SK.get_cublasHandle(), nnodes, &omega, w.data(), 1) )

    return;
}

Multigrid_GPU::Multigrid_GPU( Multigrid const & source )
:nlev(source.Nlevels()), ndofs(source.Ndofs()), _vSK(), _u(), _f(), _d(), _w(), _meshes( source._meshes )
{
    cout << "\n........................  in Multigrid::Multigrid_GPU  ..................\n";
    // Allocate Memory for matrices/vectors on all levels
    for (int lev = 0; lev < Nlevels(); ++lev) {
        _vSK.emplace_back( source._vSK[lev] );  // CRS matrix
        _u.emplace_back( source._u[lev] );
        _f.emplace_back( source._f[lev] );
        _d.emplace_back( source._d[lev] );
        _w.emplace_back( source._w[lev] );
    }
    // Intergrid transfer operators
    _vPc2f.emplace_back( ); // no prolongation to the coarsest grid
    for (int lev = 1; lev < Nlevels(); ++lev) {
        _vPc2f.emplace_back( source._vPc2f[lev] );
        //checkInterpolation(lev);
        //checkRestriction(lev);
    }
    cout << "\n..........................................\n";
}

Multigrid_GPU::~Multigrid_GPU()
{}

void Multigrid_GPU::JacobiSolve(int lev)
{
    assert(lev < Nlevels());
    ::JacobiSolve_GPU(_vSK[lev], _f[lev], _u[lev]);
}

void Multigrid_GPU::MG_Step(int lev, int const pre_smooth, bool const bzero, int nu)
{
    assert(lev < Nlevels());
    int const post_smooth = pre_smooth;

    if (lev == 0) { // coarse level
        // GH: a factorization (once in setup) with repeated forward-backward substitution would be better
        int n_jacobi_iterations = GetMesh(lev).Nnodes()/10; // ensure accuracy for coarse grid solver
        //cout << "n_jacobi: " << n_jacobi_iterations << endl;
        //cout << "n_jacobi_iterations: " << n_jacobi_iterations << endl;
        tic();
        JacobiSmoother_GPU(_vSK[lev], _f[lev], _u[lev], _d[lev],  n_jacobi_iterations, 1.0, true);
        cout << "     coarse grid solver [sec]: " << toc() << endl;
    }
    else {
        JacobiSmoother_GPU(_vSK[lev], _f[lev], _u[lev], _d[lev],  pre_smooth, 0.85, bzero || lev < Nlevels()-1);
        if (nu > 0) {
            _vSK[lev].Defect(_d[lev], _f[lev], _u[lev]);   //   d := f - K*u
            _vPc2f[lev].MultT(_d[lev], _f[lev - 1]);       // f_H := R*d

        /*
        double sk_d, sk_u, sk_f, sk_w;
        int display_level = lev;
        CHECK_CUBLAS( cublasDdot(_vSK[display_level].get_cublasHandle(), _u[display_level].size(), _u[display_level].data(), 1, _u[display_level].data(), 1, &sk_u) ) 
        CHECK_CUBLAS( cublasDdot(_vSK[display_level].get_cublasHandle(), _d[display_level].size(), _d[display_level].data(), 1, _d[display_level].data(), 1, &sk_d) ) 
        CHECK_CUBLAS( cublasDdot(_vSK[display_level].get_cublasHandle(), _f[display_level].size(), _f[display_level].data(), 1, _f[display_level].data(), 1, &sk_f) ) 
        CHECK_CUBLAS( cublasDdot(_vSK[display_level].get_cublasHandle(), _w[display_level].size(), _w[display_level].data(), 1, _w[display_level].data(), 1, &sk_w) ) 
        cout << "sk_GPU: " << "u: " << sk_u << ", d: " << sk_d << ", f: " << sk_f << ", w: " << sk_w << endl;
        //assert(false);
        */

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

void Multigrid_GPU::MG_Solve(int pre_smooth, double eps, int nu)
{
    int lev=Nlevels()-1;                // fine level

    double s0(-1);
    double si(0);
    // start with zero guess

    DiagPrecond_GPU(_vSK[lev], _f[lev], _u[lev], 1.0);  // w   := D^{-1]*f
    CHECK_CUBLAS( cublasDdot(_vSK[lev].get_cublasHandle(), _f[lev].size(), _f[lev].data(), 1, _u[lev].data(), 1, &s0) )    // s_0 := <f,u>
    
    bool bzero = true;                       // start with zero guess
    int  iter  = 0;
    do
    {
        MG_Step(lev, pre_smooth, bzero, nu);
        bzero=false;
        _vSK[lev].Defect(_d[lev], _f[lev], _u[lev]);    //   d := f - K*u
        DiagPrecond_GPU(_vSK[lev], _d[lev], _w[lev], 1.0);  // w   := D^{-1]*d
        cublasDdot(_vSK[lev].get_cublasHandle(), _d[lev].size(), _d[lev].data(), 1, _w[lev].data(), 1, &si);       // s_i := <d,w>
        ++iter;
        cout << "\nrel. error: " << sqrt(si/s0) << "  ( " << iter << " iter.)" << endl;
    } while (si>s0*eps*eps && iter<1000);


    cout << "\nrel. error: " << sqrt(si/s0) << "  ( " << iter << " iter.)" << endl;

    return;
}

[[maybe_unused]] bool Multigrid_GPU::checkInterpolation(int const lev)
{
    assert(1<=lev && lev<Nlevels());

    vector<double> wup(_w[lev].size() );
    vector<double> wdown( _w[lev-1].size());
    vector<double> d(_d[lev].size());
    cudaMemcpy( wup.data(), _w[lev].data(), wup.size() * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy( wdown.data(), _w[lev-1].data(), wdown.size() * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy( d.data(), _d[lev].data(), d.size() * sizeof(double), cudaMemcpyDeviceToHost );

    _meshes[lev-1].SetValues(wdown, [](double x, double y) -> double
                           {    double EPS = 1e-3;
                                if( abs( x ) < EPS || abs( x - 1 ) < EPS || abs( y ) < EPS || abs( y - 1 ) < EPS )
                                    return 0;
                                return x+y; }  );
    _meshes[lev].SetValues(wup, [](double /*x*/, double /*y*/) -> double
                           { return -123.0; }  );
    //static_cast<BisectInterpolation>(_vPc2f[lev]).Mult(_d[lev], _w[lev - 1]);        // d := P*w_H

    cudaMemcpy( _w[lev-1].data(), wdown.data(), wdown.size() * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( _d[lev].data(), d.data(), d.size() * sizeof(double), cudaMemcpyHostToDevice );

    _vPc2f[lev].Mult(_d[lev], _w[lev - 1]);        // d := P*w_H

    cudaMemcpy( d.data(), _d[lev].data(), d.size() * sizeof(double), cudaMemcpyDeviceToHost );
    
    cout << "înterpolated  " << endl; GetMesh(lev).Visualize(d);
    
    return true;
}

[[maybe_unused]] bool Multigrid_GPU::checkRestriction(int const lev)
{
    assert(1<=lev && lev<Nlevels());

    vector<double> wdown( _w[lev-1].size());
    vector<double> d(_d[lev].size());
    cudaMemcpy( wdown.data(), _w[lev-1].data(), wdown.size() * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy( d.data(), _d[lev].data(), d.size() * sizeof(double), cudaMemcpyDeviceToHost );

    _meshes[lev].SetValues(d, [](double x, double y) -> double
                           {    double EPS = 1e-3;
                                if( abs( x ) < EPS || abs( x - 1 ) < EPS || abs( y ) < EPS || abs( y - 1 ) < EPS )
                                    return 0;
                                return x + y; }  );
    _meshes[lev-1].SetValues(wdown, [](double /*x*/, double /*y*/) -> double
                           { return -123.0; }  );
    //static_cast<BisectInterpolation>(_vPc2f[lev]).MultT(_d[lev], _w[lev - 1]);        // w_H := R*d

    cudaMemcpy( _w[lev-1].data(), wdown.data(), wdown.size() * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( _d[lev].data(), d.data(), d.size() * sizeof(double), cudaMemcpyHostToDevice );


    _vPc2f[lev].MultT(_d[lev], _w[lev - 1]);        // w_H := R*d
    
    cudaMemcpy( wdown.data(), _w[lev-1].data(), wdown.size() * sizeof(double), cudaMemcpyDeviceToHost );

    cout << "restricted  " << endl; GetMesh(lev-1).Visualize(wdown);
    
    return true;    
}
