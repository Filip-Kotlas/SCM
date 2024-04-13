#include "mylib.h"
#include <cassert>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;


vector<double> MatVec(vector<vector<double>> const &A, vector<double> const &u)
{
    int const nrows = static_cast<int>(A.size());         // #matrix rows
    int const mcols = static_cast<int>(A[0].size());      // #matrix columns
    assert( mcols ==  static_cast<int>(u.size()) );       // check compatibility inner dimensions

    vector<double> f(nrows);                 // allocate resulting vector

                                             // matrix times vector: f := A * u
    for (int i = 0; i < nrows; ++i) {
        double tmp = 0.0;			         // initialize f[i]
        for (int j = 0; j < mcols; ++j) {
            tmp = tmp + A[i][j] * u[j];
        }
        f[i] = tmp;
        //cout << A[i].data() << endl;           // Address of a[i][0]
    }

    return f;
}
// ---------------------------------------------------------------------

vector<double> MatVec(vector<double> const &A, vector<double> const &u)
{
    int const nelem = static_cast<int>(A.size());      // #elements in matrix
    int const mcols = static_cast<int>(u.size());      // #elements in vector <==> #columns in matrix

    assert(nelem % mcols == 0);                        // nelem has to be a multiple of mcols (==> #rows)
    int const nrows = nelem/mcols;                     // integer division!

    vector<double> f(nrows);                 // allocate resulting vector
                                             // matrix times vector: f := A * u
    for (int i = 0; i < nrows; ++i) {
        double tmp = 0.0;			         // initialize f[i]
        for (int j = 0; j < mcols; ++j) {
            tmp = tmp + A[i*mcols+j] * u[j];
        }
        f[i] = tmp;
        //cout << A[i*mcols].data() << endl;           // Address of a[i][0]
    }

    return f;
}

DenseMatrix::DenseMatrix( int n, int m )
{
    int nm = max(n, m);
    data = vector<double>( n * m );
    double x_i = 0;
    double x_j = 0;
    for( int i = 0; i < n; i++ )
    {
        x_i = (double)(10 * i) / ( nm - 1 ) - 5;
        for( int j = 0; j < m; j++ )
        {
            x_j = (double)(10 * j) / ( nm - 1 ) - 5;
            data.at( i * m + j ) = 1 / ( (1 + exp( -x_i ) ) * (1 + exp( -x_j ) ) );
        }
    }
}

vector<double> DenseMatrix::Mult( const vector<double>& u ) const
{
    int const nelem = static_cast<int>(data.size());
    int const mcols = static_cast<int>(u.size());

    assert(nelem % mcols == 0);
    int const nrows = nelem/mcols;

    vector<double> output( nrows );
    double temp = 0;
    for ( int i = 0; i < nrows; i++)
    {
        for( int j = 0; j < mcols; j++ )
        {
            temp += data.at(i * mcols + j ) * u.at(j);
        }
        output.at(i) = temp;
        temp = 0;
    }
    return output;
}

vector<double> DenseMatrix::MultT( const vector<double>& u ) const
{
    int const nelem = static_cast<int>(data.size());
    int const nrows= static_cast<int>(u.size());

    assert(nelem % nrows == 0);
    int const mcols = nelem/nrows;

    vector<double> output( mcols );
    double temp = 0;
    for ( int i = 0; i < mcols; i++)
    {
        for( int j = 0; j < nrows; j++ )
        {
            temp += data.at( j * mcols + i) * u.at(j);
        }
        output.at(i) = temp;
        temp = 0;
    }
    return output;
}

ostream& operator<<( ostream& os, const DenseMatrix& dt )
{
    for( size_t i = 0; i < dt.data.size(); i++ )
    {
        os << dt.data.at(i) << ", ";
    }
    os << endl;
    return os;
}

MatrixFromVectors::MatrixFromVectors( vector<double> u_, vector<double> v_ )
: u(u_), v(v_)
{
}

MatrixFromVectors::MatrixFromVectors( size_t n, size_t m )
{
    u = vector<double>(n);
    v = vector<double>(m);
    double x_i = 0;
    for( int i = 0; i < n; i++ )
    {
        x_i = (double) (10 * i) / ( n - 1 ) - 5;
        u.at(i) = 1 / (1 + exp( -x_i ) );
    }
    
    for( int i = 0; i < m; i++ )
    {
        x_i = (double) (10 * i) / ( m - 1 ) - 5;
        v.at(i) = 1 / (1 + exp( -x_i ) );
    }
}

vector<double> MatrixFromVectors::Mult( const vector<double>&  w ) const
{
    assert( v.size() == w.size() );
    vector<double> result( u.size() );
    double temp = 0;

    for( int i = 0; i < w.size(); i++ )
    {
        temp += w.at(i) * v.at(i);
    }

    for( int i = 0; i < u.size(); i++ )
    {
        result.at(i) = temp * u.at(i);
    }
    return result;
}

vector<double> MatrixFromVectors::MultT( const vector<double>& w ) const
{
    assert( u.size() == w.size() );
    vector<double> result( v.size() );
    double temp = 0;

    for( int i = 0; i < w.size(); i++ )
    {
        temp += w.at(i) * u.at(i);
    }

    for( int i = 0; i < v.size(); i++ )
    {
        result.at(i) = temp * v.at(i);
    }
    return result;
}

ostream& operator<<(std::ostream& os, const MatrixFromVectors& dt)
{
    os << "u: ";
    for( int i = 0; i < dt.u.size(); i++ )
    {
        os << dt.u.at(i) << " ";
    }
    os << endl;

    os << "v: ";
    for( int i = 0; i < dt.v.size(); i++ )
    {
        os << dt.v.at(i) << " ";
    }
    os << endl;


}