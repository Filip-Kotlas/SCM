#include "matrix.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <cuda_runtime.h>

void print_first_5_elements( const float * vector )
{
    for( int i = 0; i < 5; i++ )
    {
        std::cout << vector[i] << ", ";
    }
    std::cout << " ..." << std::endl;
}

int main ()
{
    cublasStatus_t  stat;
    cublasHandle_t  handle;

    std::chrono::time_point<std::chrono::system_clock> t1, t2;
    std::chrono::duration<double, std::milli>  time_span;


    const int M = 100;      //can't be smaller than 5
    const int N = 100;

    stat = cublasCreate (& handle );
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "ERROR: cublasInit() failed!" << std::endl;
        exit(1);
    }

    float *x_h = new float [N];
    float *y_h = new float [N];
    float *z_h = new float [N];
    float alpha = 2.0;
    float beta = 3.0;
    
    for (int i = 0; i < N; i++)
    {
        x_h[i] = (i % 11);
        y_h[i] = 11 - (i % 11);
        z_h[i] = 0;
    }

    float *x_d, *y_d, *z_d; // device data
    cudaMalloc((void **) &x_d, N * sizeof(float));
    cudaMalloc((void **) &y_d, N * sizeof(float));
    cudaMalloc((void **) &z_d, N * sizeof(float));

    cublasSetVector(N, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(N, sizeof(float), y_h, 1, y_d, 1);
    cublasSetVector(N, sizeof(float), z_h, 1, z_d, 1);

    std::cout << "First five elements of the vectors x and y are: " << std::endl;
    std::cout << "x = ( ";
    print_first_5_elements(x_h);
    std::cout << "y = ( ";
    print_first_5_elements(y_h);
    std::cout << "After each calculation, they are reset to this state." << std::endl << std::endl;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSaxpy(handle, N, &alpha, x_d, 1, y_d, 1);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(y_h, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    time_span = t2 - t1;
    std::cout << "y = " << alpha <<"*x + y :" << std::endl << "y = ( ";
    print_first_5_elements(y_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //_______________________________________________________________________

    for (int i = 0; i < N; i++)
    {
        x_h[i] = (i % 11);
        y_h[i] = 11 - (i % 11);
    }
    cublasSetVector(N, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(N, sizeof(float), y_h, 1, y_d, 1);
    const float dummy = 1;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSscal(handle, N, &alpha, x_d, 1);
    cublasSaxpy( handle, N, &dummy, y_d, 1, x_d, 1);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(x_h, x_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    time_span = t2 - t1;
    std::cout << "x = " << alpha <<"*x + y :" << std::endl << "x = ( ";
    print_first_5_elements(x_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;

    //_________________________________________________________________________________
    
    for (int i = 0; i < N; i++)
    {
        x_h[i] = (i % 11);
        y_h[i] = 11 - (i % 11);
    }
    cublasSetVector(N, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(N, sizeof(float), y_h, 1, y_d, 1);

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSscal(handle, N, &alpha, x_d, 1);
    cublasSscal(handle, N, &beta, y_d, 1);
    cublasSaxpy(handle, N, &dummy, x_d, 1, y_d, 1);
    cublasSswap(handle, N, z_d, 1, y_d, 1);   
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(z_h, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    time_span = t2 - t1;
    std::cout << "z = " << alpha <<"*x + " << beta << "*y :" << std::endl << "z = ( ";
    print_first_5_elements(z_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //__________________________________________________________________________________

    for (int i = 0; i < N; i++)
    {
        x_h[i] = (i % 11);
        y_h[i] = 11 - (i % 11);
    }
    cublasSetVector(N, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(N, sizeof(float), y_h, 1, y_d, 1);
    float dot_product = 0;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSdot(handle, N, x_d, 1, y_d, 1, &dot_product);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time_span = t2 - t1;
    std::cout << "<x, y> :" << std::endl << "<x, y> = " << dot_product << std::endl;
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //__________________________________________________________________________________

    for (int i = 0; i < N; i++)
    {
        x_h[i] = (i % 11);
    }
    cublasSetVector(N, sizeof(float), x_h, 1, x_d, 1);
    float norm = 0;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSnrm2(handle, N, x_d, 1, &norm );
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time_span = t2 - t1;
    std::cout << "||x|| :" << std::endl << "||x|| = " << norm << std::endl;
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //__________________________________________________________________________________

    


    delete [] x_h;
    delete [] y_h;
    cudaFree(x_d);
    cudaFree(y_d);
    cublasDestroy(handle );
}