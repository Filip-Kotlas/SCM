#include "binaryIO.h"
#include "crsmatrix.h"
#include "precond.h"
#include "vdop.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <list>
#include <string>
#include <vector>
#include <omp.h>
using namespace std;

CRS_Matrix::CRS_Matrix()
    : _nrows(0), _ncols(0), _id(0), _ik(0), _sk(0)
{}

CRS_Matrix::CRS_Matrix(const std::string &filename) : _nrows(0), _ncols(0),  _id(0), _ik(0), _sk(0)
{
    readBinary(filename);
    _nrows = static_cast<int>(size(_id) - 1);
    _ncols = _nrows;
}


CRS_Matrix::~CRS_Matrix()
{}

void CRS_Matrix::Mult(vector<double> &w, vector<double> const &u) const
{
    assert( _ncols == static_cast<int>(u.size()) ); // compatibility of inner dimensions
    assert( _nrows == static_cast<int>(w.size()) ); // compatibility of outer dimensions

    #pragma omp parallel for
    for (int row = 0; row < _nrows; ++row)
    {
        double wi = 0.0;
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij)
        {
            wi += _sk[ij] * u[ _ik[ij] ];
        }
        w[row] = wi;
    }
    return;
}

void CRS_Matrix::Defect(vector<double> &w,
                        vector<double> const &f, vector<double> const &u) const
{
    assert( _ncols == static_cast<int>(u.size()) ); // compatibility of inner dimensions
    assert( _nrows == static_cast<int>(w.size()) ); // compatibility of outer dimensions
    assert( w.size() == f.size() );

    #pragma omp parallel for
    for (int row = 0; row < _nrows; ++row)
    {
        double wi = f[row];
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij)
        {
            wi -= _sk[ij] * u[ _ik[ij] ];
        }
        w[row] = wi;
    }
    return;
}

// solves K\f  -> u
void CRS_Matrix::cg(std::vector<double> &u, std::vector<double> const &f, 
               int const max_iterations, double const eps) const
{
	assert(_ncols == _nrows);
    assert( _ncols == static_cast<int>(u.size()) ); // compatibility of inner dimensions
    assert( _nrows == static_cast<int>(f.size()) ); // compatibility of outer dimensions
    
    vector<double> r(size(f));
    vector<double> w(size(u));
    vector<double> s(size(u));
    vector<double> v(size(f));
    vector<double> D(size(u)); GetDiag(D);        // diag(K)
    
    fill(begin(u),end(u),0.0);                    // u = 0
    Defect(r,f,u);                                // r = f-K*u
    
    vddiv(w,r,D);                                 // w = D^(-1)*r
    //w=r;
    s = w;                                        // s = w
    
    double sigma = dscapr(w,r);                   // sig = <w,r>
    double sigma0(sigma);
    int    iter(0);
    cout << iter << " iterations : error " << sqrt(sigma) << "  rel. error: " << sqrt(sigma/sigma0) << endl;
  
    while (sigma0*eps*eps<sigma && iter<max_iterations)
    {
		++iter;
		double sig_old(sigma);
		Mult(v,s);                                // v = K*s
		double alpha = sigma/dscapr(s,v);         // alf = sig/<s,v>
		vdaxpy(u,u,alpha,s);                      // u = u+alf*s
		vdaxpy(r,r,-alpha,v);                     // r = r-alf*v
		vddiv(w,r,D);                             // w = D^(-1)*r
        //w=r;
		sigma = dscapr(w,r);                      // sig = <w,r>
		     //cout << iter << "  sigma  " << sigma << endl;        
		double beta = sigma/sig_old;
		vdaxpy(s,w,beta,s);                       // s = w+beta*s
	}
    cout << iter << " iterations : error " << sqrt(sigma) << "  rel. error: " << sqrt(sigma/sigma0) << endl;
}

void CRS_Matrix::GetDiag(vector<double> &d) const
{
    // be carefull when using a rectangular matrix
    int const nm = min(_nrows, _ncols);

    assert( nm == static_cast<int>(d.size()) ); // instead of stopping we could resize d and warn the user

    #pragma omp parallel for
    for (int row = 0; row < nm; ++row)
    {
        const int ia = fetch(row, row); // Find diagonal entry of row
        assert(ia >= 0);
        d[row] = _sk[ia];
    }
    cout << ">>>>> CRS_Matrix::GetDiag  <<<<<" << endl;
    return;
}

inline
int CRS_Matrix::fetch(int const row, int const col) const
{
    int const id2 = _id[row + 1];    // end   and
    int       ip  = _id[row];        // start of recent row (global index)

    while (ip < id2 && _ik[ip] != col)   // find index col (global index)
    {
        ++ip;
    }
    if (ip >= id2)
    {
        ip = -1;
#ifndef NDEBUG                 // compiler option -DNDEBUG switches off the check
        cout << "No column  " << col << "  in row  " << row << endl;
        assert(ip >= id2);
#endif
    }
    return ip;
}

void CRS_Matrix::Debug() const
{
//  ID points to first entry of row
//  no symmetry assumed
    cout << "\nMatrix  (" << _nrows << " x " << _ncols << "  with  nnz = " << _id[_nrows] << ")\n";

    for (int row = 0; row < _nrows; ++row)
    {
        cout << "Row " << row << " : ";
        int const id1 = _id[row];
        int const id2 = _id[row + 1];
        for (int j = id1; j < id2; ++j)
        {
            cout.setf(ios::right, ios::adjustfield);
            cout << "[" << setw(2) << _ik[j] << "]  " << setw(4) << _sk[j] << "  ";
        }
        cout << endl;
    }
    return;
}

void CRS_Matrix::first3values() const
{
	cout << "\nid on CPU " << _id[0] << " " << _id[1] << " " << _id[2] << endl;
	cout << "\nik on CPU " << _ik[0] << " " << _ik[1] << " " << _ik[2] << endl;
	cout << "\nsk on CPU " << _sk[0] << " " << _sk[1] << " " << _sk[2] << endl;
}

void CRS_Matrix::writeBinary(const std::string &file) const
{
    vector<int> cnt(size(_id) - 1);
    for (size_t k = 0; k < size(cnt); ++k)
    {
        cnt[k] = _id[k + 1] - _id[k];
    }
    //adjacent_difference( cbegin(_id)+1, cend(_id), cnt );
    write_binMatrix(file, cnt, _ik, _sk);
}

void CRS_Matrix::readBinary(const std::string &file)
{
    vector<int> cnt;
    read_binMatrix(file, cnt, _ik, _sk);
    _id.resize(size(cnt) + 1);
    _id[0] = 0;
    for (size_t k = 0; k < size(cnt); ++k)
    {
        _id[k + 1] = _id[k] + cnt[k];
    }
    //partial_sum( cbegin(cnt), cend(cnt), begin(_id)+1 );
}


void CRS_Matrix::getNumberOffDiagonals(int& klow, int& kup) const
{
    klow=0;                                  // #subdiagonals  
    kup =0;                                  // #superdiagonals
    //  Assumption: column indices ordered ascending 
    for (int row=0; row<_nrows; ++row)
    {
        int const clow = _ik[_id[row]];       // first column index
        int const cup  = _ik[_id[row+1]-1];   // last  column index
        klow = max(klow,row-clow);
        kup  = max( kup,cup-row);
    }
}

// https://www.netlib.org/lapack/lug/node124.html
//#define what_is(x) cerr << #x << " is \n" << x << endl;
void CRS_Matrix::getBandMatrix4LapackLU
    (int& nrows, int& ncols, int& klow, int& kup, 
     vector<double>& AB, int& ldab) const
{
    nrows=_nrows;
    ncols=_ncols;
    getNumberOffDiagonals(klow, kup);
    //ldab = klow+kup+1;                  // only band matrix
    ldab = klow+(klow+kup)+1;             // band matrix for LU factorization
    what_is(nrows) what_is(klow) what_is(kup) what_is(ldab) 
    assert(ldab<=nrows);
   
    AB.resize(static_cast<long long int>(ldab)*ncols,0.0);  // size > 2^31
    assert(AB.size()/ldab==ncols);

    for (int i=0; i<_nrows; ++i)             // row index
    {
        for (int ij=_id[i]; ij<_id[i+1]; ++ij)
        {
            long long int const j=_ik[ij];             // column index   
            //AB.at(j*ldab+(ldab-klow)+i-j-1) = _sk[ij]; 
            AB[j*ldab+(ldab-klow)+i-j-1] = _sk[ij]; 
        }
    }
}                        
                        
                        
                        
                        
                        
