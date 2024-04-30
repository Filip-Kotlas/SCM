//Filip Kotlas
//task 2 subtask 1, 3, 5

// originates from Ruetsch/Oster: Getting Started with CUDA
// more C++-style by Haase
#include <cassert>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>

__global__ void inc_gpu(float *const a, int N);

__global__ void inc_gpu(float *const a, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        a[idx] = a[idx] + 1;
}

__global__ void sum_gpu(float *const a, float *const b, float *const c, int N);
__global__ void sum_gpu(float *const a, float *const b, float *const c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void log_gpu(float *const arg, float *const res, int N);
__global__ void log_gpu(float *const arg, float *const res, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        res[idx] = log(arg[idx]);
    }
}

__global__ void exp_gpu(float *const arg, float *const res, int N);
__global__ void exp_gpu(float *const arg, float *const res, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        res[idx] = exp(arg[idx]);
    }
}

using namespace std;

int main(void)
{
    cout << setprecision(10);
    int const N = 10000;
    int const nBytes = N * sizeof(float);

    int const blockSize = 64;
    int const numBlocks = (N + blockSize - 1) / blockSize;

    chrono::time_point<std::chrono::system_clock> t1, t2;
    chrono::duration<double, std::milli>  time_span;

    float *a_h = new float [N];     // host data
    float *b_h = new float [N];     // host data
    float *a_d, *b_d;               // device data
    cudaMalloc((void **) &a_d, nBytes);
    cudaMalloc((void **) &b_d, nBytes);    

    for (int i = 0; i < N; i++) a_h[i] = 100.0f + static_cast<float>(i);

    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);    //  a_d <- a_h
    cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);  //  b_d <- a_d

    // ---------------------------------------------------------
// Manipulate on GPU
    t1 = chrono::high_resolution_clock::now();
    inc_gpu <<< numBlocks, blockSize>>>(b_d, N);             //  b_d := b_d+1.0
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();
    
    cout << cudaGetErrorName(cudaGetLastError()) << endl;
    
    cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);    //  b_h <- b_d

// Check on CPU
    for (int i = 0; i < N; i++) assert( a_h[i] == b_h[i] - 1.0f );
    cout << "Check 1  OK" << endl;
    time_span = t2 - t1;
    cout << "Incrementing: The ellapsed time is " << time_span.count() << " milliseconds." << endl;
// ---------------------------------------------------------

    delete [] b_h;
    delete [] a_h;
    cudaFree(a_d);
    cudaFree(b_d);
    cout << endl;

//_______________________________________________________________
// My extension of the code
    
    //Task 2, subtask 1
    float *summand_1_h = new float[N];
    float *summand_2_h = new float[N];
    float *result_h = new float[N];
    float *summand_1_d, *summand_2_d, *result_d;
    cudaMalloc((void **) &summand_1_d, nBytes);
    cudaMalloc((void **) &summand_2_d, nBytes);
    cudaMalloc((void **) &result_d, nBytes);

    for( int i = 0; i < N; i++ )
    {
        summand_1_h[i] = static_cast<float>(i);
        summand_2_h[i] = static_cast<float>(N) - static_cast<float>(i);
    }

    cudaMemcpy(summand_1_d, summand_1_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(summand_2_d, summand_2_h, nBytes, cudaMemcpyHostToDevice);

    t1 = chrono::high_resolution_clock::now();
    sum_gpu <<< numBlocks, blockSize >>> (summand_1_d, summand_2_d, result_d, N );
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();
    cout << cudaGetErrorName(cudaGetLastError()) << endl;

    cudaMemcpy( result_h, result_d, nBytes, cudaMemcpyDeviceToHost);

    bool success = true;
    for( int i = 0; i < N; i++)
    {
        if( result_h[i] != N )
        {
            cout << "Addition: Instead of " << N << " there is " << result_h[i] << " on position " << i << "." << endl;
            success = false;
        }
    }
    if(success)
        cout << "Vectors were successfuly added together." << endl;

    time_span = t2 - t1;
    cout << "Summation: The ellapsed time is " << time_span.count() << " milliseconds." << endl;


    delete [] summand_1_h;
    delete [] summand_2_h;
    delete [] result_h;
    cudaFree(summand_1_d);
    cudaFree(summand_2_d);
    cudaFree(result_d);
    cout << endl;

    //Task 2, subtask 3
    float *argument_h = new float[N];
    float *after_exponentiation_h = new float[N];
    float *argument_d, *after_logarithm_d, *after_exponentiation_d;
    cudaMalloc((void **) &argument_d, nBytes);
    cudaMalloc((void **) &after_logarithm_d, nBytes);
    cudaMalloc((void **) &after_exponentiation_d, nBytes);

    for( int i = 0; i < N; i++ )
    {
        argument_h[i] = static_cast<float>(i);
    }

    cudaMemcpy(argument_d, argument_h, nBytes, cudaMemcpyHostToDevice);

    t1 = chrono::high_resolution_clock::now();
    log_gpu <<< numBlocks, blockSize >>> (argument_d, after_logarithm_d, N );
    exp_gpu <<< numBlocks, blockSize >>> (after_logarithm_d, after_exponentiation_d, N );
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();

    cout << cudaGetErrorName(cudaGetLastError()) << endl;

    cudaMemcpy( after_exponentiation_h, after_exponentiation_d, nBytes, cudaMemcpyDeviceToHost);

    success = true;
    float precision = 1e-1;
    for( int i = 0; i < N; i++)
    {
        if( abs(after_exponentiation_h[i] - argument_h[i]) > precision )
        {
            cout << abs(after_exponentiation_h[i] - argument_h[i]) << endl;
            cout << "Log & exp: Instead of " << argument_h[i] << " we get " << after_exponentiation_h[i] << " on position " << i << "." << endl;
            success = false;
        }
    }
    if(success)
        cout << "Log and exp where successfully applied to the vector. The results agree with an accuracy of " << precision << "." << endl;
    
    time_span = t2 - t1;
    cout << "Log & exp: The ellapsed time is " << time_span.count() << " milliseconds." << endl;

    delete [] argument_h;
    delete [] after_exponentiation_h;
    cudaFree(argument_d);
    cudaFree(after_logarithm_d);
    cudaFree(after_exponentiation_d);
    cout << endl;

    return 0;
}
