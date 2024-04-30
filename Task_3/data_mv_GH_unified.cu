//Filip Kotlas
//task 2 subtask 4

// originates from Ruetsch/Oster: Getting Started with CUDA
// more C++-style by Haase
#include <cassert>
#include <iostream>
#include <cmath>
#include <iomanip>

__global__ void inc_gpu(float *const a, int N);

__global__ void inc_gpu(float *const a, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        a[idx] = a[idx] + 1;
}

__global__ void sum_gpu(float *const a, float *const b, float* const c, int N);
__global__ void sum_gpu(float *const a, float *const b, float* const c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void log_gpu(float* arg, float* res, int N);
__global__ void log_gpu(float* arg, float* res, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        res[idx] = log(arg[idx]);
    }
}

__global__ void exp_gpu(float* arg, float* res, int N);
__global__ void exp_gpu(float* arg, float* res, int N)
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_span = 0;

    float *a, *b;                    // device data
    cudaMallocManaged(&a, nBytes);
    cudaMallocManaged(&b, nBytes);

    for (int i = 0; i < N; i++) a[i] = 100.0f + static_cast<float>(i);

    cudaMemcpy(b, a, nBytes, cudaMemcpyDeviceToDevice);  //  b <- a

// ---------------------------------------------------------
// Manipulate on GPU
    cudaEventRecord(start);
    inc_gpu <<< numBlocks, blockSize>>>(b, N);             //  b := b+1.0
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

    for( int i = 0; i < N; i++ )
    {
        summand_1[i] = static_cast<float>(i);
        summand_2[i] = static_cast<float>(N) - static_cast<float>(i);
    }

    cudaEventRecord(start);
    sum_gpu <<< numBlocks, blockSize >>> (summand_1, summand_2, result, N );
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

    for( int i = 0; i < N; i++ )
    {
        argument[i] = static_cast<float>(i);
    }

    cudaEventRecord(start);
    log_gpu <<< numBlocks, blockSize >>> (argument, after_logarithm, N );
    exp_gpu <<< numBlocks, blockSize >>> (after_logarithm, after_exponentiation, N );
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
