#include "matrix.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <cuda_runtime.h>
#include "equal_reduction.h"

void print_first_5_elements( const float * vector )
{
    for( int i = 0; i < 5; i++ )
    {
        std::cout << vector[i] << ", ";
    }
    std::cout << " ..." << std::endl;
}

void print_matrix( denseMatrix<float> M )
{
    std::cout << std::endl;
    for( int i = 0; i < M.GetNrows(); i++ )
    {
        for( int j = 0; j < M.GetNcols(); j++ )
        {
            std::cout << M[i + M.GetNrows() * j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_matrix( float* M, int n, int m)
{
    std::cout << std::endl;
    for( int i = 0; i < n; i++ )
    {
        for( int j = 0; j < m; j++ )
        {
            std::cout << M[i + n * j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
int main ()
{
    cublasStatus_t  stat;
    cublasHandle_t  handle;

    std::chrono::time_point<std::chrono::system_clock> t1, t2;
    std::chrono::duration<double, std::milli>  time_span;

    const int n = 10000;
    const int m = 10000;      //can't be smaller than 5

    denseMatrix<float> M(n, m);

    std::cout << "n = " << n << std::endl << "m = " << m << std::endl << std::endl;

    //Setting up unit matrix with element (1, 2) equal 1 for testing.
    /*
    for( int i = 0; i < n; i++ )
    {
        for( int j = 0; j < m; j++ )
        {
            if( i == j )
            {
                M[j*n + i] = 1;
            }
            else
            {
                M[j*n + i] = 0;
            }
        }
    }
    M[1*n] = 1;
    */

    stat = cublasCreate (& handle );
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "ERROR: cublasInit() failed!" << std::endl;
        exit(1);
    }

    float *x_h = new float [n];
    float *y_h = new float [n];
    float *z_h = new float [n];
    float alpha = 2.0;
    float beta = 3.0;
    
    for (int i = 0; i < n; i++)
    {
        x_h[i] = (i % 11);
        y_h[i] = 11 - (i % 11);
        z_h[i] = 0;
    }

    float *x_d, *y_d, *z_d; // device data
    cudaMalloc((void **) &x_d, n * sizeof(float));
    cudaMalloc((void **) &y_d, n * sizeof(float));
    cudaMalloc((void **) &z_d, n * sizeof(float));

    cublasSetVector(n, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(n, sizeof(float), y_h, 1, y_d, 1);
    cublasSetVector(n, sizeof(float), z_h, 1, z_d, 1);

    std::cout << "First five elements of the vectors x and y are: " << std::endl;
    std::cout << "x = ( ";
    print_first_5_elements(x_h);
    std::cout << "y = ( ";
    print_first_5_elements(y_h);
    std::cout << "After each calculation in the first part, they are reset to this state." << std::endl << std::endl;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSaxpy(handle, n, &alpha, x_d, 1, y_d, 1);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(y_h, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    time_span = t2 - t1;
    std::cout << "y = " << alpha <<"*x + y :" << std::endl << "y = ( ";
    print_first_5_elements(y_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //_______________________________________________________________________

    for (int i = 0; i < n; i++)
    {
        y_h[i] = 11 - (i % 11);
    }
    cublasSetVector(n, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(n, sizeof(float), y_h, 1, y_d, 1);
    const float one = 1;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSscal(handle, n, &alpha, x_d, 1);
    cublasSaxpy( handle, n, &one, y_d, 1, x_d, 1);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(x_h, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    time_span = t2 - t1;
    std::cout << "x = " << alpha <<"*x + y :" << std::endl << "x = ( ";
    print_first_5_elements(x_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;

    //_________________________________________________________________________________
    
    for (int i = 0; i < n; i++)
    {
        x_h[i] = (i % 11);
    }
    cublasSetVector(n, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(n, sizeof(float), y_h, 1, y_d, 1);

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSscal(handle, n, &alpha, x_d, 1);
    cublasSscal(handle, n, &beta, y_d, 1);
    cublasSaxpy(handle, n, &one, x_d, 1, y_d, 1);
    cublasSswap(handle, n, z_d, 1, y_d, 1);   
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(z_h, z_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    time_span = t2 - t1;
    std::cout << "z = " << alpha <<"*x + " << beta << "*y :" << std::endl << "z = ( ";
    print_first_5_elements(z_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //__________________________________________________________________________________

    cublasSetVector(n, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(n, sizeof(float), y_h, 1, y_d, 1);
    float dot_product = 0;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSdot(handle, n, x_d, 1, y_d, 1, &dot_product);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time_span = t2 - t1;
    std::cout << "<x, y> :" << std::endl << "<x, y> = " << dot_product << std::endl;
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //__________________________________________________________________________________

    float norm = 0;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSnrm2(handle, n, x_d, 1, &norm );
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();

    time_span = t2 - t1;
    std::cout << "||x|| :" << std::endl << "||x|| = " << norm << std::endl;
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //__________________________________________________________________________________

    std::cout << "-------------------------------------" << std::endl;

    delete [] x_h;
    cudaFree(x_d);

    float* r_h = new float[n];
    x_h = new float[m];

    for (int i = 0; i < m; i++)
        x_h[i] = pow( -1, i );

    for( int i = 0; i < n; i++ )
        r_h[i] = 0;

    float* r_d;
    float* M_data_d;
    cudaMalloc((void **) &r_d, n * sizeof(float));
    cudaMalloc((void **) &x_d, m * sizeof(float));
    cudaMalloc((void **) &M_data_d, m * n * sizeof(float));    

    cublasSetVector(m, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(n, sizeof(float), r_h, 1, r_d, 1);
    cublasSetVector(m*n, sizeof(float), M.data(), 1, M_data_d, 1);

    float zero = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N, n, m, &one, M_data_d, n, x_d, 1, &zero, r_d, 1); //dummy call

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSgemv(handle, CUBLAS_OP_N, n, m, &one, M_data_d, n, x_d, 1, &zero, r_d, 1);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(r_h, r_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    //print_matrix(M);

    time_span = t2 - t1;
    std::cout << "r = M*x" << std::endl << "r = ( ";
    print_first_5_elements(r_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //__________________________________________________________________________________

    delete [] x_h;
    cudaFree(x_d);
    delete [] r_h;
    cudaFree(r_d);

    r_h = new float[m];
    x_h = new float[n];

    for (int i = 0; i < n; i++)
        x_h[i] = 1;//pow( -1, i );

    for( int i = 0; i < m; i++ )
        r_h[i] = 0;

    cudaMalloc((void **) &r_d, m * sizeof(float));
    cudaMalloc((void **) &x_d, n * sizeof(float));

    cublasSetVector(n, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(m, sizeof(float), r_h, 1, r_d, 1);

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSgemv(handle, CUBLAS_OP_T, n, m, &one, M_data_d, n, x_d, 1, &zero, r_d, 1);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(r_h, r_d, m * sizeof(float), cudaMemcpyDeviceToHost);

    time_span = t2 - t1;
    std::cout << "r = trans(M)*x" << std::endl << "r = ( ";
    print_first_5_elements(r_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    //__________________________________________________________________________________

    std::cout << "-------------------------------------" << std::endl;

    delete [] x_h;
    cudaFree(x_d);
    delete [] r_h;
    cudaFree(r_d);


    float* T_h = new float[3*n];
    x_h = new float[n];
    r_h = new float[n];

    for( int i = 0; i < n; i++ )
    {
        T_h[3*i] = -1;  
        T_h[3*i + 1] = 2;
        T_h[3*i + 2] = -1;
    }
    T_h[0] = 0;
    T_h[3*n-1] = 0;

    for( int i = 0; i < n; i++)
        x_h[i] = 1;

    for( int i = 0; i < n; i++ )
        r_h[i] = 0;

    float* T_d;
    cudaMalloc((void **) &T_d,3 * n * sizeof(float));
    cudaMalloc((void **) &x_d, n * sizeof(float));
    cudaMalloc((void **) &r_d, n * sizeof(float));

    cublasSetVector(3 * n, sizeof(float), T_h, 1, T_d, 1);
    cublasSetVector(n, sizeof(float), x_h, 1, x_d, 1);
    cublasSetVector(n, sizeof(float), r_h, 1, r_d, 1);

    cudaDeviceSynchronize();
    cublasSgbmv(handle, CUBLAS_OP_N, n, n, 1, 1, &one, T_d, 3, x_d, 1, &zero, r_d, 1); //dummy call

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSgbmv(handle, CUBLAS_OP_N, n, n, 1, 1, &one, T_d, 3, x_d, 1, &zero, r_d, 1);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(r_h, r_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    time_span = t2 - t1;
    std::cout << "r = T*x" << std::endl << "r = ( ";
    print_first_5_elements(r_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
        
    //__________________________________________________________________________________
    
    std::cout << "-------------------------------------" << std::endl;

    delete [] x_h;
    cudaFree(x_d);
    delete [] y_h;
    cudaFree(y_d);
    delete [] z_h;
    cudaFree(z_d);

    float* A_h = new float[n*n];
    x_h = new float[n];
    y_h = new float[n];
    z_h = new float[n];
    float* temp_h = new float[m];

    for( int i = 0; i < n; i++ )
    {
        x_h[i] = 1e-5;
        y_h[i] = 0;
        z_h[i] = 0;
    }

    float *A_d;
    float *temp_d;
    cudaMalloc((void**) &A_d, n * n * sizeof(float) );
    cudaMalloc((void**) &y_d, n * sizeof(float) );
    cudaMalloc((void**) &z_d, n * sizeof(float) );
    cudaMalloc((void**) &temp_d, n * sizeof(float) );

    cublasSetVector( n, sizeof(float), x_h, 1, x_d, 1);

    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m, &one, M_data_d, n, M_data_d, n, &zero, A_d, n ); //dummy call

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m, &one, M_data_d, n, M_data_d, n, &zero, A_d, n );
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(A_h, A_d, n * n * sizeof(float), cudaMemcpyDeviceToHost );

    time_span = t2 - t1;
    std::cout << "A = M*trans(M)" << std::endl;
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSgemv(handle, CUBLAS_OP_N, n, n, &one, A_d, n, x_d, 1, &zero, z_d, 1 );
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(z_h, z_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    
    time_span = t2 - t1;
    std::cout << "z = A*x" << std::endl << "z = ( ";
    print_first_5_elements(z_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;
    
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    cublasSgemv(handle, CUBLAS_OP_T, n, m, &one, M_data_d, n, x_d, 1, &zero, temp_d, 1 );
    cublasSgemv(handle, CUBLAS_OP_N, n, m, &one, M_data_d, n, temp_d, 1, &zero, y_d, 1 );
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(y_h, y_d, n * sizeof(float), cudaMemcpyDeviceToHost );
    
    time_span = t2 - t1;
    std::cout << "y = M*(trans(M)*x)" << std::endl << "y = ( ";
    print_first_5_elements(y_h);
    std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;

    float eps = 1e-3;
    bool equal = are_equal(y_d, z_d, n, eps );
    std::cout << "Vectors y and z are " << (equal ? "equal" : "not equal") << " with precision of" << eps << "." <<  std::endl << std::endl;

    //__________________________________________________________________________________

    std::cout << "-------------------------------------" << std::endl;

    if( n != m )
    {
        std::cout << "Can't extract diagonal from non-square matrix." << std::endl;
    }
    else
    {
        float* diag_h = new float[n];
        float* diag_d;
        cudaMalloc( (void**) &diag_d, n * sizeof(float) );

        cudaDeviceSynchronize();
        t1 = std::chrono::high_resolution_clock::now();
        cublasScopy(handle, n, M_data_d, n + 1, diag_d, 1);
        cudaDeviceSynchronize();
        t2 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(diag_h, diag_d, n * sizeof(float), cudaMemcpyDeviceToHost );

        time_span = t2 - t1;
        std::cout << "y = diag(M)" << std::endl << "y = ( ";
        print_first_5_elements(diag_h);
        std::cout << "Time: " << time_span.count() << " milliseconds." << std::endl << std::endl;

        delete [] diag_h;
        cudaFree(diag_d);
    }


    //__________________________________________________________________________________

    delete [] x_h;
    delete [] y_h;
    delete [] z_h;
    delete [] r_h;
    delete [] T_h;
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaFree(r_d);
    cudaFree(M_data_d);
    cudaFree(T_d);
    cublasDestroy(handle);
}