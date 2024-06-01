#include "binaryIO.h"
#include "getmatrix.h"
#include "userset.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>                  // contains clock()
#include <iomanip>
#include <iostream>
#include <list>
#include <string>
#include <utility>
#include <vector>
using namespace std;

// ####################################################################

Matrix::Matrix(int const nrows, int const ncols)
    : _nrows(nrows), _ncols(ncols), _dd(0)
{}

Matrix::~Matrix()
{}

// ####################################################################

CRS_Matrix::CRS_Matrix()
    : Matrix(0, 0), _nnz(0), _id(0), _ik(0), _sk(0)
{}

CRS_Matrix::CRS_Matrix(const std::string &filename) : Matrix(0, 0), _nnz(0), _id(0), _ik(0), _sk(0)
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

    for (int row = 0; row < _nrows; ++row) {
        double wi = 0.0;
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij) {
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

    for (int row = 0; row < _nrows; ++row) {
        double wi = f[row];
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij) {
            wi -= _sk[ij] * u[ _ik[ij] ];
        }
        w[row] = wi;
    }
    return;
}


void CRS_Matrix::JacobiSmoother(std::vector<double> const &f, std::vector<double> &u,
                    std::vector<double> &r, int nsmooth, double const omega, bool zero) const
{
    // ToDO: ensure compatible dimensions
    assert(_ncols==_nrows);
    assert( _ncols == static_cast<int>(u.size()) ); // compatibility of inner dimensions
    assert( _nrows == static_cast<int>(r.size()) ); // compatibility of outer dimensions
    assert( r.size() == f.size() );
    
    auto const &D = Matrix::GetDiag();        // accumulated diagonal of matrix @p SK.
         
    if (zero) {            // assumes initial solution is zero
        for (int k = 0; k < _nrows; ++k) {
            // u := u + om*D^{-1}*f
            u[k] = omega*f[k] / D[k]; // MPI: distributed to accumulated vector needed
        }
        --nsmooth;                           // first smoothing sweep done
    }

 
    for (int ns = 1; ns <= nsmooth; ++ns) {
        for (int row = 0; row < _nrows; ++row) {
            double wi = f[row];
            for (int ij = _id[row]; ij < _id[row + 1]; ++ij) {
                wi -= _sk[ij] * u[ _ik[ij] ];
            }
            r[row] = wi;
        }        
        for (int k = 0; k < _nrows; ++k) {
            // u := u + om*D^{-1}*r
            u[k] = u[k] + omega * r[k] / D[k]; // MPI: distributed to accumulated vector needed
        }
    }
    return;
}

void CRS_Matrix::GetDiag(vector<double> &d) const
{
    // be carefull when using a rectangular matrix
    int const nm = min(_nrows, _ncols);
    assert( nm == static_cast<int>(d.size()) ); // instead of stopping we could resize d and warn the user

    for (int row = 0; row < nm; ++row) {
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

    while (ip < id2 && _ik[ip] != col) { // find index col (global index)
        ++ip;
    }
    if (ip >= id2) {
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

    for (int row = 0; row < _nrows; ++row) {
        cout << "Row " << row << " : ";
        int const id1 = _id[row];
        int const id2 = _id[row + 1];
        for (int j = id1; j < id2; ++j) {
            cout.setf(ios::right, ios::adjustfield);
            cout << "[" << setw(2) << _ik[j] << "]  " << setw(4) << _sk[j] << "  ";
        }
        cout << endl;
    }
    return;
}



bool CRS_Matrix::Compare2Old(int nnode, int const id[], int const ik[], double const sk[]) const
{
    bool bn = (nnode == _nrows);     // number of rows
    if (!bn) {
        cout << "#########   Error: " << "number of rows" << endl;
    }

    bool bz = (id[nnode] == _nnz);   // number of non zero elements
    if (!bz) {
        cout << "#########   Error: " << "number of non zero elements" << endl;
    }

    bool bd = equal(id, id + nnode + 1, _id.cbegin()); // row starts
    if (!bd) {
        cout << "#########   Error: " << "row starts" << endl;
    }

    bool bk = equal(ik, ik + id[nnode], _ik.cbegin()); // column indices
    if (!bk) {
        cout << "#########   Error: " << "column indices" << endl;
    }

    bool bv = equal(sk, sk + id[nnode], _sk.cbegin()); // values
    if (!bv) {
        cout << "#########   Error: " << "values" << endl;
    }

    return bn && bz && bd && bk && bv;
}


void CRS_Matrix::writeBinary(const std::string &file)
{
    vector<int> cnt(size(_id) - 1);
    for (size_t k = 0; k < size(cnt); ++k) {
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
    for (size_t k = 0; k < size(cnt); ++k) {
        _id[k + 1] = _id[k] + cnt[k];
    }
    //partial_sum( cbegin(cnt), cend(cnt), begin(_id)+1 );
}

// ####################################################################

FEM_Matrix::FEM_Matrix(Mesh const &mesh)
    : CRS_Matrix(), _mesh(mesh)
{
    Derive_Matrix_Pattern();
    return;
}

FEM_Matrix::~FEM_Matrix()
{}

void FEM_Matrix::Derive_Matrix_Pattern_fast()
{
    cout << "\n############   FEM_Matrix::Derive_Matrix_Pattern ";
    MyTimer tstart;    //tstart.tic();
    
    int const nelem(_mesh.Nelems());
    int const ndof_e(_mesh.NdofsElement());
    auto const &ia(_mesh.GetConnectivity());
//  Determine the number of matrix rows
    _nrows = *max_element(ia.cbegin(), ia.cbegin() + ndof_e * nelem);
    ++_nrows;                                 // node numberng: 0 ... nnode-1
    assert(*min_element(ia.cbegin(), ia.cbegin() + ndof_e * nelem) == 0); // numbering starts with 0 ?

// CSR data allocation
    _id.resize(_nrows + 1);                  // Allocate memory for CSR row pointer
//##########################################################################
    auto const v2v = _mesh.Node2NodeGraph();
    _nnz = 0;                         // number of connections
    _id[0] = 0;                       // start of matrix row zero
    for (size_t v = 0; v < v2v.size(); ++v ) {
        auto nv=static_cast<int>(v2v[v].size());
        _id[v + 1] = _id[v] + nv;
        _nnz += nv;
    }
    assert(_nnz == _id[_nrows]);
    _sk.resize(_nnz);                        // Allocate memory for CSR column index vector

// CSR data allocation
    _ik.resize(_nnz);                        // Allocate memory for CSR column index vector
// Copy column indices
    int kk = 0;
    for (const auto & v : v2v) {
        for (size_t vi = 0; vi < v.size(); ++vi) {
            _ik[kk] = v[vi];
            ++kk;
        }
    }
    _ncols = *max_element(_ik.cbegin(), _ik.cend());  // maximal column number
    ++_ncols;                                         // node numbering: 0 ... nnode-1
    //cout << _nrows << "  " << _ncols << endl;
    assert(_ncols == _nrows);

    cout << "finished in  " <<  tstart.toc()  << " sec.    ########\n";
  

    return;
}


void FEM_Matrix::Derive_Matrix_Pattern_slow()
{
    cout << "\n############   FEM_Matrix::Derive_Matrix_Pattern slow ";
    auto tstart = clock();
    int const nelem(_mesh.Nelems());
    int const ndof_e(_mesh.NdofsElement());
    auto const &ia(_mesh.GetConnectivity());
//  Determine the number of matrix rows
    _nrows = *max_element(ia.cbegin(), ia.cbegin() + ndof_e * nelem);
    ++_nrows;                                 // node numberng: 0 ... nnode-1
    assert(*min_element(ia.cbegin(), ia.cbegin() + ndof_e * nelem) == 0); // numbering starts with 0 ?

//  Collect for each node those nodes it is connected to (multiple entries)
//  Detect the neighboring nodes
    vector< list<int> > cc(_nrows);             //  cc[i] is the  list of nodes a node 'i' is connected to
    for (int i = 0; i < nelem; ++i) {
        int const idx = ndof_e * i;
        for (int k = 0; k < ndof_e; ++k) {
            list<int> &cck = cc[ia[idx + k]];
            cck.insert( cck.end(), ia.cbegin() + idx, ia.cbegin() + idx + ndof_e );
        }
    }
//  Delete the multiple entries
    _nnz = 0;
    for (auto &it : cc) {
        it.sort();
        it.unique();
        _nnz += static_cast<int>(it.size());
        // cout << it.size() << " :: "; copy(it->begin(),it->end(), ostream_iterator<int,char>(cout,"  ")); cout << endl;
    }

// CSR data allocation
    _id.resize(_nrows + 1);                  // Allocate memory for CSR row pointer
    _ik.resize(_nnz);                        // Allocate memory for CSR column index vector

//  copy CSR data
    _id[0] = 0;                              // begin of first row
    for (size_t i = 0; i < cc.size(); ++i) {
        //cout << i << "   " << nid.at(i) << endl;;
        const list<int> &ci = cc[i];
        const auto nci = static_cast<int>(ci.size());
        _id[i + 1] = _id[i] + nci; // begin of next line
        copy(ci.begin(), ci.end(), _ik.begin() + _id[i] );
    }

    assert(_nnz == _id[_nrows]);
    _sk.resize(_nnz);                        // Allocate memory for CSR column index vector

    _ncols = *max_element(_ik.cbegin(), _ik.cend());  // maximal column number
    ++_ncols;                                 // node numbering: 0 ... nnode-1
    //cout << _nrows << "  " << _ncols << endl;
    assert(_ncols == _nrows);

    double duration = static_cast<double>(clock() - tstart) / CLOCKS_PER_SEC;  // ToDo: change to  systemclock
    cout << "finished in  " <<  duration  << " sec.    ########\n";

    return;
}


void FEM_Matrix::CalculateLaplace(vector<double> &f)
{
    cout << "\n############   FEM_Matrix::CalculateLaplace ";
    assert(_mesh.NdofsElement() == 3);               // only for triangular, linear elements
    assert(_nnz == _id[_nrows]);

    for (int k = 0; k < _nrows; ++k) {
        _sk[k] = 0.0;
    }
    for (int k = 0; k < _nrows; ++k) {
        f[k] = 0.0;
    }

    double ske[3][3], fe[3];
    //  Loop over all elements
    auto const nelem = _mesh.Nelems();
    auto const &ia   = _mesh.GetConnectivity();
    auto const &xc   = _mesh.GetCoords();

    for (int i = 0; i < nelem; ++i) {
        CalcElem(ia.data() + 3 * i, xc.data(), ske, fe);
        AddElem_3(ia.data() + 3 * i, ske, fe, f);
    }
    return;
}


void FEM_Matrix::ApplyDirichletBC(std::vector<double> const &u, std::vector<double> &f)
{
    auto const idx = _mesh.Index_DirichletNodes();
    int const nidx = static_cast<int>(idx.size());

    for (int i = 0; i < nidx; ++i) {
        int const row = idx[i];
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij) {
            int const col = _ik[ij];
            if (col == row) {
                _sk[ij] = 1.0;
                f[row]  = u[row];
            }
            else {
                int const id1 = fetch(col, row); // Find entry (col,row)
                assert(id1 >= 0);
                f[col] -= _sk[id1] * u[row];
                _sk[id1] = 0.0;
                _sk[ij]  = 0.0;
            }
        }
    }

    return;
}



void FEM_Matrix::AddElem_3(int const ial[3], double const ske[3][3], double const fe[3], vector<double> &f)
{
    for (int i = 0; i < 3; ++i) {
        const int ii  = ial[i];           // row ii (global index)
        for (int j = 0; j < 3; ++j) {     // no symmetry assumed
            const int jj = ial[j];        // column jj (global index)
            const int ip = fetch(ii, jj);       // find column entry jj in row ii
#ifndef NDEBUG                 // compiler option -DNDEBUG switches off the check
            if (ip < 0) {      // no entry found !!
                cout << "Error in AddElem: (" << ii << "," << jj << ") ["
                     << ial[0] << "," << ial[1] << "," << ial[2] << "]\n";
                assert(ip >= 0);
            }
#endif
            _sk[ip] += ske[i][j];
        }
        f[ii] += fe[i];
    }
}

bool CRS_Matrix::CheckSymmetry() const
{
    cout << "+++  Check matrix symmetry  +++" << endl;
    bool bs{true};
    for (int row = 0; row < Nrows(); ++row) {
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij) {
            const int col = _ik[ij];       // column col (global index)
            const int ip = fetch(col, row);  // find column entry row in row col
            if (ip < 0) {      // no entry found !!
                cout << "Matrix has non-symmetric pattern at (" << row << "," << col << ")" << endl;
                bs = false;
                //assert(ip >= 0);
            }
            if ( std::abs(_sk[ij] - _sk[ip]) > 1e-13) {
                cout << "Matrix has non-symmetric entries at (" << row << "," << col << ")" << endl;
                bs = false;
            }
        }
    }
    return bs;
}


bool CRS_Matrix::CheckRowSum() const
{
    cout << "+++  Check row sum  +++" << endl;
    vector<double> rhs(Ncols(), 1.0);
    vector<double> res(Nrows());

    Mult(res, rhs);

    bool bb{true};
    for (size_t k = 0; k < res.size(); ++k) {
        //if (std::abs(res[k]) != 0.0)
        if (std::abs(res[k]) > 1e-14) {
            cout << "!! Nonzero row " << k << " : sum = " << res[k] << endl;
            bb = false;
        }
    }
    return bb;
}

bool CRS_Matrix::CheckMproperty() const
{
    cout << "+++  Check M property  +++" << endl;
    bool bm{true};
    for (int row = 0; row < Nrows(); ++row) {
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij) {
            if (_ik[ij] == row) {
                bool b_diag = _sk[ij] > 0.0;
                if (!b_diag) {
                    cout << "## negative diag in row " << row << " : " << _sk[ij] << endl;
                    bm = false;
                }
            }
            else {
                bool b_off = _sk[ij] <= 0.0;
                if (!b_off) {
                    //cout << "!! positive off-diag [" << row << "," << _ik[ij] << "] : " << _sk[ij] << endl;
                    bm = false;
                }
            }
        }
    }
    return bm;
}

bool CRS_Matrix::ForceMproperty()
{
    cout << "+++  Force M property  +++" << endl;
    bool bm{false};
    for (int row = 0; row < Nrows(); ++row) {
        double corr{0.0};
        int idiag = {-1};
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij) {
            if (_ik[ij] != row &&  _sk[ij] > 0.0) {
                corr   += _sk[ij];
                _sk[ij] = 0.0;
                bm = true;
            }
            if (_ik[ij] == row) {
                idiag = ij;
            }
        }
        assert(idiag >= 0);
        _sk[idiag] += corr;
    }
    return bm;
}

bool CRS_Matrix::CheckMatrix() const
{
    bool b0 = CheckSymmetry();
    if (!b0) {
        cout << " !!!!  N O   S Y M M E T R Y" << endl;
    }
    
    bool b1 = CheckRowSum();
    if (!b1) {
        cout << " !!!!  R O W   S U M   E R R O R" << endl;
    }

    bool b2 = CheckMproperty();
    if (!b2) {
        cout << " !!!!  N O   M - M A T R I X" << endl;
    }

    return b1 && b2;
}

void CRS_Matrix::GetDiag_M(vector<double> &d) const
{
    // be carefull when using a rectangular matrix
    //int const nm = min(_nrows, _ncols);
    assert( min(_nrows, _ncols) == static_cast<int>(d.size()) ); // instead of stopping we could resize d and warn the user

    for (int row = 0; row < Nrows(); ++row) {
        d[row] = 0.0;
        double v_ii{-1.0};
        for (int ij = _id[row]; ij < _id[row + 1]; ++ij) {
            if (_ik[ij] != row) {
                d[row] += std::abs(_sk[ij]);
            }
            else {
                v_ii = _sk[ij];
            }
        }
        if ( d[row] < v_ii ) {
            d[row] = v_ii;
        }
    }
    cout << "<<<<<<<  GetDiag_M (finished)   >>>>>>>>>" << endl;
    return;
}


//  general routine for lin. triangular elements

void CalcElem(int const ial[3], double const xc[], double ske[3][3], double fe[3])
//void CalcElem(const int* __restrict__ ial, const double* __restrict__ xc, double* __restrict__ ske[3], double* __restrict__ fe)
{
    const int  i1  = 2 * ial[0],   i2 = 2 * ial[1],   i3 = 2 * ial[2];
    const double x13 = xc[i3 + 0] - xc[i1 + 0],  y13 = xc[i3 + 1] - xc[i1 + 1],
                 x21 = xc[i1 + 0] - xc[i2 + 0],  y21 = xc[i1 + 1] - xc[i2 + 1],
                 x32 = xc[i2 + 0] - xc[i3 + 0],  y32 = xc[i2 + 1] - xc[i3 + 1];
    const double jac = fabs(x21 * y13 - x13 * y21);

    ske[0][0] = 0.5 / jac * (y32 * y32 + x32 * x32);
    ske[0][1] = 0.5 / jac * (y13 * y32 + x13 * x32);
    ske[0][2] = 0.5 / jac * (y21 * y32 + x21 * x32);
    ske[1][0] = ske[0][1];
    ske[1][1] = 0.5 / jac * (y13 * y13 + x13 * x13);
    ske[1][2] = 0.5 / jac * (y21 * y13 + x21 * x13);
    ske[2][0] = ske[0][2];
    ske[2][1] = ske[1][2];
    ske[2][2] = 0.5 / jac * (y21 * y21 + x21 * x21);

    const double xm    = (xc[i1 + 0] + xc[i2 + 0] + xc[i3 + 0]) / 3.0,
                 ym    = (xc[i1 + 1] + xc[i2 + 1] + xc[i3 + 1]) / 3.0;
    //fe[0] = fe[1] = fe[2] = 0.5 * jac * FunctF(xm, ym) / 3.0;
    fe[0] = fe[1] = fe[2] = 0.5 * jac * fNice(xm, ym) / 3.0;
}

void CalcElem_Masse(int const ial[3], double const xc[], double ske[3][3])
{
    const int  i1  = 2 * ial[0],   i2 = 2 * ial[1],   i3 = 2 * ial[2];
    const double x13 = xc[i3 + 0] - xc[i1 + 0],  y13 = xc[i3 + 1] - xc[i1 + 1],
                 x21 = xc[i1 + 0] - xc[i2 + 0],  y21 = xc[i1 + 1] - xc[i2 + 1];
    //x32 = xc[i2 + 0] - xc[i3 + 0],  y32 = xc[i2 + 1] - xc[i3 + 1];
    const double jac = fabs(x21 * y13 - x13 * y21);

    ske[0][0] += jac / 12.0;
    ske[0][1] += jac / 24.0;
    ske[0][2] += jac / 24.0;
    ske[1][0] += jac / 24.0;
    ske[1][1] += jac / 12.0;
    ske[1][2] += jac / 24.0;
    ske[2][0] += jac / 24.0;
    ske[2][1] += jac / 24.0;
    ske[2][2] += jac / 12.0;

    return;
}

// #####################################################################

BisectInterpolation::BisectInterpolation()
    : Matrix( 0, 0 ), _iv(), _vv()
{
}

BisectInterpolation::BisectInterpolation(std::vector<int> const &fathers)
    : Matrix( static_cast<int>(fathers.size()) / 2, 1 + * max_element(fathers.cbegin(), fathers.cend()) ),
      _iv(fathers), _vv(fathers.size(), 0.5)
{
}

BisectInterpolation::~BisectInterpolation()
{}

void BisectInterpolation::GetDiag(vector<double> &d) const
{
    assert( Nrows() == static_cast<int>(d.size()) );

    for (int k = 0; k < Nrows(); ++k) {
        if ( _iv[2 * k] == _iv[2 * k + 1] ) {
            d[k] = 1.0;
        }
        else {
            d[k] = 0.0;
        }
    }
    return;
}

void BisectInterpolation::Mult(vector<double> &wf, vector<double> const &uc) const
{
    assert( Nrows() == static_cast<int>(wf.size()) );
    assert( Ncols() == static_cast<int>(uc.size()) );

    for (int k = 0; k < Nrows(); ++k) {
        wf[k] = _vv[2 * k] * uc[_iv[2 * k]] + _vv[2 * k + 1] * uc[_iv[2 * k + 1]];
    }
    return;
}

void BisectInterpolation::MultT(vector<double> const &wf, vector<double> &uc) const
{
    assert(  Nrows() == static_cast<int>( wf.size()) );
    assert(  Ncols() == static_cast<int>( uc.size()) );
    assert(2*Nrows() == static_cast<int>(_iv.size()) );
    assert(2*Nrows() == static_cast<int>(_vv.size()) );
    
    for (int k = 0; k < Ncols(); ++k)  uc[k] = 0.0;
   
// GH: atomic slows down the code ==> use different storage for MultT operation (CRS-matrix?)
    for (int k = 0; k < Nrows(); ++k) {
        int const j1=_iv[2 * k    ];
        int const j2=_iv[2 * k + 1];
        uc[j1] += _vv[2 * k  ] * wf[k];
        uc[j2] += _vv[2 * k + 1] * wf[k];
    }
    return;
}


void BisectInterpolation::MultT_Full(vector<double> const &wf, vector<double> &uc) const
{
    assert( Nrows() == static_cast<int>(wf.size()) );
    assert( Ncols() == static_cast<int>(uc.size()) );
// GH: atomic slows down the code ==> use different storage for MultT operation (CRS-matrix?)
    for (int k = 0; k < Ncols(); ++k)  uc[k] = 0.0;
    vector<double> full(uc.size(),0.0);
    for (int k = 0; k < Nrows(); ++k) {
        if (_iv[2 * k] != _iv[2 * k + 1]) {
            uc[_iv[2 * k]  ] += _vv[2 * k  ] * wf[k];
            uc[_iv[2 * k + 1]] += _vv[2 * k + 1] * wf[k];
            full[_iv[2 * k    ]] += _vv[2 * k  ];
            full[_iv[2 * k + 1]] += _vv[2 * k + 1];
        }
        else {
            uc[_iv[2 * k]  ] +=  2.0*_vv[2 * k  ] * wf[k]; // uses a property of class BisectInterpolation
            full[_iv[2 * k] ] += 2.0*_vv[2 * k  ];
        }
    }
    for (size_t k=0; k<uc.size(); ++k)  uc[k] /= full[k];
    return;
}

void BisectInterpolation::Defect(vector<double> &w,
                                 vector<double> const &f, vector<double> const &u) const
{
    assert( Nrows() == static_cast<int>(w.size()) );
    assert( Ncols() == static_cast<int>(u.size()) );
    assert( w.size() == f.size() );

    for (int k = 0; k < Nrows(); ++k) {
        w[k] = f[k] - _vv[2 * k] * u[_iv[2 * k]] + _vv[2 * k + 1] * u[_iv[2 * k + 1]];
    }
    return;
}

void BisectInterpolation::Debug() const
{
    for (int k = 0; k < Nrows(); ++k) {
        cout << k << " : fathers(" << _iv[2 * k] << "," << _iv[2 * k + 1] << ")    ";
        cout << "weights(" << _vv[2 * k] << "," << _vv[2 * k + 1] << endl;
    }
    cout << endl;
    return;
}

int BisectInterpolation::fetch(int row, int col) const
{
    int idx(-1);
    if (_iv[2 * row  ] == col) idx = 2 * row;
    if (_iv[2 * row + 1] == col) idx = 2 * row + 1;
    assert(idx >= 0);
    return idx;
}

// #####################################################################

BisectIntDirichlet::BisectIntDirichlet(std::vector<int> const &fathers, std::vector<int> idxc_dir)
    : BisectInterpolation(fathers), _idxDir(std::move(idxc_dir))
{
}

BisectIntDirichlet::~BisectIntDirichlet()
{}


void BisectIntDirichlet::MultT(vector<double> const &wf, vector<double> &uc) const
{
    BisectInterpolation::MultT(wf, uc);
    for (int kc : _idxDir) {
        uc.at(kc) = 0.0;                // Set Dirichlet node on coarse mesh to Zero
    }

    return;
}

// #####################################################################

void DefectRestrict(CRS_Matrix const &SK, BisectInterpolation const &P,
                    vector<double> &fc, vector<double> &ff, vector<double> &uf)
{
    assert( P.Nrows() == static_cast<int>(ff.size()) );
    assert( P.Ncols() == static_cast<int>(fc.size()) );
    assert( ff.size() == uf.size() );
    assert( P.Nrows() == SK.Nrows() );

    for (int k = 0; k < P.Ncols(); ++k)  fc[k] = 0.0;

// GH: atomic slows down the code ==> use different storage for MultT operation (CRS-matrix?)
    for (int row = 0; row < SK._nrows; ++row) {
        double wi = ff[row];
        for (int ij = SK._id[row]; ij < SK._id[row + 1]; ++ij) {
            wi -= SK._sk[ij] * uf[ SK._ik[ij] ];
        }

        const int i1 = P._iv[2 * row];
        const int i2 = P._iv[2 * row + 1];
        fc[i1] += P._vv[2 * row  ] * wi;
        fc[i2] += P._vv[2 * row + 1] * wi;
    }
    return;
}
