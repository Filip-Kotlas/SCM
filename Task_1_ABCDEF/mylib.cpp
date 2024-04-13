#include "mylib.h"
#include <cassert>       // assert()
#include <cmath>
#include <vector>
#include <list>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "mayer_primes.h"

#ifdef __INTEL_CLANG_COMPILER
#pragma message(" ##########  Use of MKL  ###############")
#include <mkl.h>
#else
#pragma message(" ##########  Use of CBLAS  ###############")
//extern "C"
//{
#include <cblas.h>               // cBLAS Library
#include <lapacke.h>             // Lapack
//}
#endif

using namespace std;

double scalar(vector<double> const &x, vector<double> const &y)
{
    assert(x.size() == y.size()); // switch off via compile flag: -DNDEBUG
    size_t const N = x.size();
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        sum += x[i] * y[i];
        //sum += exp(x[i])*log(y[i]);
    }
    return sum;
}


double scalar_cblas(vector<double> const &x, vector<double> const &y)
{
    int const asize = static_cast<int>(size(x));
    int const bsize = static_cast<int>(size(y));
    assert(asize == bsize); // switch off via compile flag: -DNDEBUG
	return cblas_ddot(asize,x.data(),1,y.data(),1);    
    //assert(x.size() == y.size()); // switch off via compile flag: -DNDEBUG
	//return cblas_ddot(x.size(),x.data(),1,y.data(),1);
}

float scalar_cblas(vector<float> const &x, vector<float> const &y)
{
    int const asize = static_cast<int>(size(x));
    int const bsize = static_cast<int>(size(y));
    assert(asize == bsize); // switch off via compile flag: -DNDEBUG
	return cblas_sdot(asize,x.data(),1,y.data(),1);    
    //assert(x.size() == y.size()); // switch off via compile flag: -DNDEBUG
	//return cblas_ddot(x.size(),x.data(),1,y.data(),1);
}


double norm(vector<double> const &x)
{
    size_t const N = x.size();
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

//Assignment A
void means(int a, int b, int c, float& arith, float& geom, float& harm)
{
    arith = (float) ( a + b + c ) / 3;
    geom = pow( a * b * c, 1.0 / 3 );
    harm = 3 / ( 1.0 / a + 1.0 / b + 1.0 / c );
}

void means( vector<double>& data, float& arith, float& geom, float& harm)
{
    arith = 0;
    geom = 1;
    harm = 0;

    for(int i : data)
    {
        arith += i;
        geom *= i;
        harm += 1.0/i;
    }

    arith /= data.size();
    geom = pow( geom, 1.0/3 );
    harm = data.size() / harm;
}

//Asignment B
float deviation( vector<double>& data )
{
    float arith = 0;
    float geom = 0;
    float harm = 0;
    float variance = 0;
    means( data, arith, geom, harm );
    
    for( int i : data )
    {
        variance += ( i - arith ) * ( i - arith );
    }
    variance /= data.size();
    return sqrt( variance );
}

//Assignment C
int summation_via_for( int n )
{
    int sum = 0;
    for( int i = 0; i <= n; i ++ )
    {
        if( i % 3 == 0 || i % 5 == 0 )
        {
            sum += i;
        }
    }
    return sum;
}

int summation_via_formula( int n )
{
    int div_3 = n / 3;
    int div_5 = n / 5;
    int div_15 = n / 15;
    int sum = 3 * div_3 * ( div_3 + 1 ) / 2 + 5 * div_5 * ( div_5 + 1 ) / 2 - 15 * div_15 * ( div_15 + 1 ) / 2;
    return sum;
}

//Assignment D
float kahan_skalar( const vector<float>& input )
{
    float sum = 0;
    float c = 0;
    for ( float i : input )
    {
        float y =  i - c;
        float t = sum + y;
        c = ( t - sum ) - y;
        sum = t;
    }
    return sum;
}

float no_kahan( const vector<float>& input )
{
    float sum = 0;
    for ( float i : input )
    {
        sum += i;
    }
    return sum;
}

void sum_of_inverse_square( vector<float>& output, int n )
{
    for ( int i = 1; i <= n; i++ )
    {
        output.push_back( 1.0/( i*i ) );
    }
}

//Assignment E
double vector_insertion( vector<int>& input )
{
    int size = input.size();
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator (seed);
    int number = 0;
    chrono::time_point<std::chrono::system_clock> t1, t2;

    t1 = chrono::high_resolution_clock::now();
    for ( int i = 0; i < size; i++ )
    {
        number = generator();
        input.insert( lower_bound( input.begin(), input.end(), number ), number );
    }
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro> duration = t2 - t1;

    return duration.count();
}

double list_insertion( list<int>& input )
{
    int size = input.size();
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator (seed);
    int number = 0;
    chrono::time_point<std::chrono::system_clock> t1, t2;

    t1 = chrono::high_resolution_clock::now();
    for ( int i = 0; i < size; i++ )
    {
        number = generator();
        input.insert( lower_bound( input.begin(), input.end(), number ), number );
    }
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro> duration = t2 - t1;

    return duration.count();
}

void measure_and_write_to_console_time( int n )
{
    vector<int> vec;
    list<int> lis;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator (seed);

    for( int i = 0; i < n; i++ )
    {
        vec.push_back( generator() );
        lis.push_back( generator() );
    }

    sort( vec.begin(), vec.end() );
    lis.sort();

    double time_vec = vector_insertion( vec );
    double time_lis = list_insertion( lis );

    cout << "Time for vector: " << (int) time_vec << " microseconds." << endl;
    cout << "Time for list:   " << (int) time_lis << " microseconds." << endl << endl;
}

//Assignment F
int single_goldbach( int n )
{
    if( n % 2 == 1 )
    {
        return 0;
    }

    int sum = 0;
    vector<int> primes = get_primes(n);
    for( int p : primes )
    {
        for( int q : primes )
        {
            if( p <= q && p + q == n )
            {
                sum++;
            }
        }
    }
    return sum;
}

vector<int> count_goldbach( int n )
{
    //get primes
    vector<int> primes = get_primes( n - 2 );

    //Erase 2 from the primes. It is not needed and we don't have to check for parity now.
    primes.erase( primes.begin() );

    vector<int> combinations( ( n - 2 ) / 2 );
    fill( combinations.begin(), combinations.end(), 0 );

    combinations.at(0) = 1;

    for( int p : primes )
    {
        for( int q : primes )
        {
            if( p <= q && p + q <= n )
            {
                combinations.at( ( p + q - 4 ) / 2 )++;
            }
        }
    }
    return combinations;
}