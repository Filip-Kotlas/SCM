//Filip Kotlas
//task 2 subtask 4

// originates from Ruetsch/Oster: Getting Started with CUDA
// more C++-style by Haase
#include <cassert>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>

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

using namespace std;

int main(void)
{
    cout << setprecision(10);
    int const N = 10000;
    int const nBytes = N * sizeof(float);

    int const blockSize = 64;
    int const numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_span = 0;

    float *a, *b;                    // device data
    cudaMallocManaged(&a, nBytes);
    cudaMallocManaged(&b, nBytes);

    thrust::sequence(a, a + N, 100.0f, 1.0f );

    cudaMemcpy(b, a, nBytes, cudaMemcpyDeviceToDevice);  //  b <- a

// ---------------------------------------------------------
// Manipulate on GPU
    cudaEventRecord(start);
    thrust::transform(thrust::cuda::par, b, b + N, b, increment());             //  b := b+1.0
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cout << cudaGetErrorName(cudaGetLastError()) << endl;
    
// Check on CPU
    for (int i = 0; i < N; i++) assert( a[i] == b[i] - 1.0f );
    cout << "Check 1  OK" << endl;
    cudaEventElapsedTime(&time_span, start, stop);
    cout << "Incrementing: The ellapsed time is " << time_span << " milliseconds." << endl;
// ---------------------------------------------------------

    cudaFree(a);
    cudaFree(b);
    cout << endl;

//_______________________________________________________________
// My extension of the code
    
    //Task 2, subtask 1
   
    float *summand_1, *summand_2, *result;
    cudaMallocManaged(&summand_1, nBytes);
    cudaMallocManaged(&summand_2, nBytes);
    cudaMallocManaged(&result, nBytes);

    thrust::sequence(summand_1, summand_1 + N, 0.0f, 1.0f);
    thrust::sequence(summand_2, summand_2 + N, static_cast<float>(N), -1.0f);

    cudaEventRecord(start);
    thrust::transform(thrust::cuda::par, summand_1, summand_1 + N, summand_2, result, thrust::plus<float>() );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << cudaGetErrorName(cudaGetLastError()) << endl;

    bool success = true;
    for( int i = 0; i < N; i++)
    {
        if( result[i] != N )
        {
            cout << "Addition: Instead of " << N << " there is " << result[i] << " on position " << i << "." << endl;
            success = false;
        }
    }
    if(success)
        cout << "Vectors were successfuly added together." << endl;

    cudaEventElapsedTime(&time_span, start, stop);
    cout << "Summation: The ellapsed time is " << time_span << " milliseconds." << endl;

    cudaFree(summand_1);
    cudaFree(summand_2);
    cudaFree(result);
    cout << endl;

    //Task 2, subtask 3
    float *argument, *after_logarithm, *after_exponentiation;
    cudaMallocManaged(&argument, nBytes);
    cudaMallocManaged(&after_logarithm, nBytes);
    cudaMallocManaged(&after_exponentiation, nBytes);

    thrust::sequence(argument, argument + N, 0.0f, 1.0f );

    cudaEventRecord(start);
    thrust::transform(thrust::cuda::par, argument, argument + N, after_logarithm, logarithm());
    thrust::transform(thrust::cuda::par, after_logarithm, after_logarithm + N, after_exponentiation, exponential());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cout << cudaGetErrorName(cudaGetLastError()) << endl;

    success = true;
    float precision = 1e-1;
    for( int i = 0; i < N; i++)
    {
        if( abs(after_exponentiation[i] - argument[i]) > precision )
        {
            cout << abs(after_exponentiation[i] - argument[i]) << endl;
            cout << "Log & exp: Instead of " << argument[i] << " we get " << after_exponentiation[i] << " on position " << i << "." << endl;
            success = false;
        }
    }
    if(success)
        cout << "Log and exp where successfully applied to the vector. The results agree with an accuracy of " << precision << "." << endl;

    cudaEventElapsedTime(&time_span, start, stop);
    cout << "Log & exp: The ellapsed time is " << time_span << " milliseconds." << endl;

    cudaFree(argument);
    cudaFree(after_logarithm);
    cudaFree(after_exponentiation);
    cout << endl;

    return 0;
}
