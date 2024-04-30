//Filip Kotlas
//task 3 subtask 3
//rewritten code of task 2 with help of thrust

// originates from Ruetsch/Oster: Getting Started with CUDA
// more C++-style by Haase
#include <cassert>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

struct increment
{
    __host__ __device__
    float operator()(const float &x)
    {
        return x + 1;
    }
};

struct logarithm
{
    __host__ __device__
    float operator()(const float &x)
    {
        return log(x);
    }
};

struct exponential
{
    __host__ __device__
    float operator()(const float &x)
    {
        return exp(x);
    }
};


int main(void)
{
    std::cout << std::setprecision(10);
    int const N = 10000;
    int const nBytes = N * sizeof(float);

    int const blockSize = 64;
    int const numBlocks = (N + blockSize - 1) / blockSize;

    std::chrono::time_point<std::chrono::system_clock> t1, t2;
    std::chrono::duration<double, std::milli>  time_span;

    thrust::host_vector<float> a_h(N);     // host data
    thrust::host_vector<float> b_h(N);     // host data
    thrust::device_vector<float> a_d(N);   // device data
    thrust::device_vector<float> b_d(N);   // device data

    thrust::sequence(a_h.begin(), a_h.end(), 100.0f, 1.0f );

    a_d = a_h;
    b_d = a_d;

// ---------------------------------------------------------
// Manipulate on GPU
    t1 = std::chrono::high_resolution_clock::now();
    thrust::transform(b_d.begin(), b_d.end(), b_d.begin(), increment());
    t2 = std::chrono::high_resolution_clock::now();

    b_h = b_d;    //  b_h <- b_d

// Check on CPU
    for (int i = 0; i < N; i++) assert( a_h[i] == b_h[i] - 1.0f );
    std::cout << "Check 1  OK" << std::endl;
    time_span = t2 - t1;
    std::cout << "Incrementing: The ellapsed time is " << time_span.count() << " milliseconds." << std::endl << std::endl;
// ---------------------------------------------------------


//_______________________________________________________________
// My extension of the code

    //Task 2, subtask 1
    thrust::host_vector<float> summand_1_h(N);
    thrust::host_vector<float> summand_2_h(N);
    thrust::host_vector<float> result_h(N);

    thrust::sequence(summand_1_h.begin(), summand_1_h.end(), 0.0f, 1.0f);
    thrust::sequence(summand_2_h.begin(), summand_2_h.end(), static_cast<float>(N), -1.0f);

    thrust::device_vector<float> summand_1_d = summand_1_h;
    thrust::device_vector<float> summand_2_d = summand_2_h;
    thrust::device_vector<float> result_d = result_h;

    t1 = std::chrono::high_resolution_clock::now();
    thrust::transform(summand_1_d.begin(), summand_1_d.end(), summand_2_d.begin(), result_d.begin(), thrust::plus<float>() );
    t2 = std::chrono::high_resolution_clock::now();

    result_h = result_d;

    bool success = true;
    for( int i = 0; i < N; i++)
    {
        if( result_h[i] != N )
        {
            std::cout << "Addition: Instead of " << N << " there is " << result_h[i] << " on position " << i << "." << std::endl;
            success = false;
        }
    }
    if(success)
        std::cout << "Vectors were successfuly added together." << std::endl;

    time_span = t2 - t1;
    std::cout << "Summation: The ellapsed time is " << time_span.count() << " milliseconds." << std::endl << std::endl;

    //Task 2, subtask 3
    thrust::host_vector<float> argument_h(N);
    thrust::host_vector<float> after_exponentiation_h(N);
    thrust::device_vector<float> argument_d(N);
    thrust::device_vector<float> after_logarithm_d(N);
    thrust::device_vector<float> after_exponentiation_d(N);

    thrust::sequence(argument_h.begin(), argument_h.end(), 0.0f, 1.0f );

    argument_d = argument_h;

    t1 = std::chrono::high_resolution_clock::now();
    thrust::transform(argument_d.begin(), argument_d.end(), after_logarithm_d.begin(), logarithm());
    thrust::transform(after_logarithm_d.begin(), after_logarithm_d.end(), after_exponentiation_d.begin(), exponential());
    t2 = std::chrono::high_resolution_clock::now();

    after_exponentiation_h = after_exponentiation_d;

    success = true;
    float precision = 1e-1;
    for( int i = 0; i < N; i++)
    {
        if( abs(after_exponentiation_h[i] - argument_h[i]) > precision )
        {
            std::cout << abs(after_exponentiation_h[i] - argument_h[i]) << std::endl;
            std::cout << "Log & exp: Instead of " << argument_h[i] << " we get " << after_exponentiation_h[i] << " on position " << i << "." << std::endl;
            success = false;
        }
    }
    if(success)
        std::cout << "Log and exp where successfully applied to the vector. The results agree with an accuracy of " << precision << "." << std::endl;
    
    time_span = t2 - t1;
    std::cout << "Log & exp: The ellapsed time is " << time_span.count() << " milliseconds." << std::endl << std::endl;

    return 0;
}
