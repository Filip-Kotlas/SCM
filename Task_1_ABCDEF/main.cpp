#include "mylib.h"
#include "file_io.h"
#include <cassert>
#include <chrono>           // timing
#include <cmath>            // sqrt()
#include <cstdlib>          // atoi()
#include <cstring>          // strncmp()
#include <ctime>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <vector>
#include <list>
#include "mayer_primes.h"
using namespace std;
using namespace std::chrono;  // timing

int main(int argc, char **argv)
{
    //Assignement A
    cout << "___________________________________" << endl;
    cout << "Assignment A:" << endl << endl;
    int a = 0;
    int b = 0;
    int c = 0;
    float arith = 0;
    float geom = 0;
    float harm = 0;
    cout.precision(6);
    cout << endl;

    a = 1;
    b = 4;
    c = 16;
    means( a, b, c, arith, geom, harm );
    cout << "Mean values of " << a << ", " << b << ", " << c << ":" << endl;
    cout << "Arithmetic: " << arith << endl;
    cout << "Geometric: " << geom << endl;
    cout << "Harmonic: " << harm << endl << endl;

    a = 2;
    b = 3;
    c = 5;
    means( a, b, c, arith, geom, harm );
    cout << "Mean values of " << a << ", " << b << ", " << c << ":" << endl;
    cout << "Arithmetic: " << arith << endl;
    cout << "Geometric: " << geom << endl;
    cout << "Harmonic: " << harm << endl << endl;

    a = 1000;
    b = 4000;
    c = 16000;
    means( a, b, c, arith, geom, harm );
    cout << "Mean values of " << a << ", " << b << ", " << c << ":" << endl;
    cout << "Arithmetic: " << arith << endl;
    cout << "Geometric: " << geom << endl;
    cout << "Harmonic: " << harm << endl;
    cout << "Obviously geometric mean is not correct because the product of the three numbers is to big for the float type." << endl;

    //Assignment B
    cout << "___________________________________" << endl;
    cout << "Assignment B: " << endl << endl;
    vector<double> data_1;
    cout << "Reading data" << endl;
    read_vector_from_file( "data_1.txt", data_1 );
    vector<double> output;
    output.push_back( *min_element( data_1.begin(), data_1.end() ) );
    output.push_back( *max_element( data_1.begin(), data_1.end() ) ) ;
    means( data_1, arith, geom, harm );
    output.push_back( arith );
    output.push_back( geom );
    output.push_back( harm );
    output.push_back( deviation( data_1 ) );

    cout << "Saving data" << endl;
    write_vector_to_file( "out_1.txt", output );

    //Assignment C
    cout << "___________________________________" << endl;
    cout << "Assignemnt C:" << endl << endl;

    cout << "The sum for n = 1001 using for loop: " << summation_via_for( 1001 ) << endl;
    cout << "The sum for n = 1001 using formula: " << summation_via_formula( 1001 ) << endl << endl;

    cout << "Measuring time for n = 1001 running 1000 times:" << endl;
    int sum = 0;
    chrono::time_point<std::chrono::system_clock> t1, t2;

    t1 = chrono::high_resolution_clock::now();
    for( int i = 0; i < 1000; i++ )
    {
        summation_via_for(1001);
    }
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro>  time_span = t2 - t1;
    cout << "Using for loop: " << time_span.count() << " microseconds" << endl;

    sum = 0;
    t1 = chrono::high_resolution_clock::now();
    for( int i = 0; i < 1000; i++ )
    {
        summation_via_formula(i);
    }
    t2 = chrono::high_resolution_clock::now();

    time_span = t2 - t1;
    cout << "Using formula: " << time_span.count() << " microseconds" << endl;

    //Assignment D
    cout << "___________________________________" << endl;
    cout << "Assignemnt D:" << endl << endl;
    
    vector<float> inverse_square_vector;
    int n = 0;
    float true_value = M_PI * M_PI / 6;

    n = 10;
    vector<int> num_iterations = {10, 1000, 10000, 50000};

    for( int i : num_iterations )
    {
        sum_of_inverse_square( inverse_square_vector, i );
        cout.precision(20);
        cout << "Sum of the serie for n = " << i << " with Kahan is:    " << kahan_skalar( inverse_square_vector ) << endl;
        cout << "Sum of the serie for n = " << i << " without Kahan is: " << no_kahan( inverse_square_vector ) << endl;
        cout.precision(10);
        cout << "Difference in the results is: " << abs( kahan_skalar( inverse_square_vector ) - no_kahan( inverse_square_vector ) ) << endl;
        cout << "Error with Kahan:    " << abs( kahan_skalar( inverse_square_vector ) - true_value ) << endl;
        cout << "Error without Kahan: " << abs( no_kahan( inverse_square_vector ) - true_value ) << endl << endl;
        inverse_square_vector.clear();

    }

    //Assignment E
    cout << "___________________________________" << endl;
    cout << "Assignemnt E:" << endl << endl;
    
    num_iterations = {10, 100, 1000, 10000};
    for(int i : num_iterations )
    {
        cout << "Times for n = " << i << ": " << endl;
        measure_and_write_to_console_time(i);
    }

    cout << "Vector is faster because it can access elements easier." << endl << endl;
    
    //Assigment F
    cout << "___________________________________" << endl;
    cout << "Assignemnt F:" << endl << endl;
    
    cout << "Number 694 has " << single_goldbach( 694 ) << " decompositions." << endl;
    cout << "Number 4 has " << single_goldbach( 4 ) << " decompositions. " << endl << endl;

    vector<int> num_decompositions = count_goldbach( 100000 );
    auto it_max = max_element( num_decompositions.begin(), num_decompositions.end() );
    cout << "For numbers lower than 100,000 the maximal number of decompositions has the number "
         << (it_max - num_decompositions.begin() ) * 2 + 4
         << " with " << *it_max << " combinations." << endl << endl;


    num_iterations = {10000, 100000, 400000, 1000000};
    for ( int i : num_iterations )
    {
        t1 = chrono::high_resolution_clock::now();
        num_decompositions = count_goldbach( i );
        t2 = chrono::high_resolution_clock::now();
        time_span = t2 - t1;
        cout << "Time to compute the combinations for n = " << i << " is " << (int) time_span.count() / 1000 << " miliseconds." << endl << endl;
    }

    return 0;
}