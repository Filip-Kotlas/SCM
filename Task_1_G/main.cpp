#include "mylib.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
using namespace std;

//int main(int argc, char** argv)
int main()
{  
    cout << "______________________" << endl
         << "Standard dense metrix:" << endl << endl;
    
    DenseMatrix const M(5,3);

    vector<double> const u{{1,2,3}};
    vector<double> f1 = M.Mult(u);

    vector<double> const v{{-1,2,-3,4,-5}};
    vector<double> f2 = M.MultT(v);

    cout << "The result of multiplication of a matrix M(5,3) and vector ( 1, 2, 3 ) is: " << endl; 
    for (size_t k = 0; k < f1.size(); k++ )
    {
        cout << f1[k] << "  ";
    }
    cout << endl;

    cout << "The result of multiplication of transpose of a matrix M(5,3) and vector ( -1, 2, -3, 4, -5 ) is: " << endl; 
    for (size_t k = 0; k < f2.size(); k++ )
    {
        cout << f2[k] << "  ";
    }
    cout << endl << endl;

    int n = 10000;
    DenseMatrix const N( n, n );
    vector<double> s;
    for( int i = 0 ; i < n; i++ )
    {
        s.push_back( pow(-1, i ) * ( (double)(i % 5) + 1 ) );
    }

    vector<double> r1( n );
    vector<double> r2( n );


    chrono::time_point<std::chrono::system_clock> t1, t2;
    t1 = chrono::high_resolution_clock::now();
    r1 = N.Mult( s );
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli>  time_span = t2 - t1;
    cout << "Standard multiplication of matrix of " << n << "*" << n << " elements and vector took " << time_span.count() << " milliseconds." << endl;

    t1 = chrono::high_resolution_clock::now();
    r2 = N.MultT( s );
    t2 = chrono::high_resolution_clock::now();
    time_span = t2 - t1;
    cout << "Multiplication of transpose of matrix of " << n << "*" << n << " elements and vector took " << time_span.count() << " milliseconds." << endl << endl;

    bool are_equal = true;
    for( int i = 0; i < n; i++ )
    {
        if( r1.at(i) != r2.at(i) )
        {
            are_equal = false;
            break;
        }
    }
    if( are_equal )
        cout << "Resulting vectors are equal." << endl;
    else
        cout << "Resulting vectors are not equal." << endl;

    cout << "Examples of elements of the vectors with the same indices are:" << endl;
    cout << "( ";
    for( int i = 0; i < n; i += n / 10 )
    {
        cout << r1.at(i) << ", ";
    }
    cout << ")" << endl;
    
    cout << "( ";
    for( int i = 0; i < n; i += n / 10  )
    {
        cout << r2.at(i) << ", ";
    }
    cout << ")" << endl;
    

    cout << "_________________________________________________" << endl
         << "Matrix defined as multiplication of two vectors: " << endl << endl;
    MatrixFromVectors H(3 , 4);
    vector<double> x = {1, 2, 1, 2};
    vector<double> y = {2, 1, 2};
    vector<double> s1 = H.Mult(x);
    vector<double> s2 = H.MultT(y);

    cout << "Product of the new type of matrix H(3, 4) with vector (1, 2, 1, 2) is: " << endl;
    for( size_t i = 0; i < s1.size(); i++ )
    {
        cout << s1.at(i) << " ";
    }
    cout << endl;

    cout << "Product of the the transpose of matrix H(3, 4) with vector (2, 1, 2) is: " << endl;
    for( size_t i = 0; i < s2.size(); i++ )
    {
        cout << s2.at(i) << " ";
    }
    cout << endl << endl;

    MatrixFromVectors const G( n, n );

    t1 = chrono::high_resolution_clock::now();
    r1 = G.Mult( s );
    t2 = chrono::high_resolution_clock::now();
    time_span = t2 - t1;
    cout << "Standard multiplication of the new type of matrix with " << n << "*" << n << " elements and a vector took " << time_span.count() << " milliseconds." << endl;

    t1 = chrono::high_resolution_clock::now();
    r2 = G.MultT( s );
    t2 = chrono::high_resolution_clock::now();
    time_span = t2 - t1;
    cout << "Multiplication of transpose of the new type of matrix with " << n << "*" << n << " elements and a vector took " << time_span.count() << " milliseconds." << endl << endl;

    are_equal = true;
    for( int i = 0; i < n; i++ )
    {
        if( r1.at(i) != r2.at(i) )
        {
            are_equal = false;
            break;
        }
    }
    if( are_equal )
        cout << "Resulting vectors are equal." << endl;
    else
        cout << "Resulting vectors are not equal." << endl;

    cout << "Examples of elements of the vectors with the same indices are:" << endl;
    cout << "( ";
    for( int i = 0; i < n; i += n / 10 )
    {
        cout << r1.at(i) << ", ";
    }
    cout << ")" << endl;
    
    cout << "( ";
    for( int i = 0; i < n; i += n / 10  )
    {
        cout << r2.at(i) << ", ";
    }
    cout << ")" << endl << endl;

    cout << "Computation time is significantly shorter for the second type of matrix." << endl;

    return 0;
}
