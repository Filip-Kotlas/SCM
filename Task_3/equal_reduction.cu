//Filip Kotlas
//task 3 subtask 1

#include "equal_reduction.h"

int main(void)
{
    float *x_h, *y_h; // vectors on host
    float *x_d, *y_d; // vectors on device

    const long int N = 100000000;
    const int nBytes = N * sizeof(float);
    const int LOOP = 10;
    const float eps = 1e-4;
    bool equal = true;

    //allocation of memory
    x_h = new float [N];
    y_h = new float [N];
    cudaMalloc((void **) &x_d, nBytes);
    cudaMalloc((void **) &y_d, nBytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_span_gpu = 0;

    //initialization of data
    for (int i = 0; i < N; i++)
    {
        x_h[i] = (i % 137) + 1;
        y_h[i] = (i % 137) + 1;
    }
    //Replace one component of vector to see wheather the code notices that the vectors don't match.
    int long index = 400 * 2 * 2 * 2 * 2;
    x_h[index] = 140;
    y_h[index] = 140;

    //copying data from host to device
    cudaMemcpy(x_d, x_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, nBytes, cudaMemcpyHostToDevice);

    equal = are_equal(x_d, y_d, N, eps );

    std::cout << "N: " << N << std::endl;
    std::cout << "LOOP: " << LOOP << std::endl;

    std::cout << std::endl << "1) Comparing vectors on gpu:" << std::endl;
    std::cout << "  a) Standard algorithm: " << std::endl;

    cudaEventRecord(start);
    for( int i = 0; i < LOOP; i++ )
    {
        equal = are_equal(x_d, y_d, N, eps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_span_gpu, start, stop);

    if(equal)
        std::cout << "    Vectors are equal." << std::endl;
    else
        std::cout << "    Vectors are not equal." << std::endl;

    std::cout << "    Computation took " << time_span_gpu << " milliseconds." << std::endl << std::endl;

    std::cout << "  b) One common variable algorithm: " << std::endl;

    dim3 dimBlock, dimGrid;
    cudaDeviceProp prop;
    cudaGetDeviceProperties (&prop, 0);
    const int blocksize = 8 * 64,
              gridsize = (N + blocksize - 1) / blocksize;
    dimBlock = dim3(blocksize);
    dimGrid  = dim3(1 * gridsize);

    bool* equal_host = new bool;
    *equal_host = true;
    bool* equal_device_global;
    cudaMalloc((void **) &equal_device_global, sizeof(bool));
    cudaMemcpy( equal_device_global, equal_host, sizeof(bool), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for( int i = 0; i < LOOP; i++ )
    {
        cudaMemcpy( equal_device_global, equal_host, sizeof(bool), cudaMemcpyHostToDevice);
        equal_with_one_common_variable<<< dimGrid, dimBlock >>>(equal_device_global, x_d, y_d, N, eps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_span_gpu, start, stop);

    cudaMemcpy( equal_host, equal_device_global,  sizeof(bool), cudaMemcpyDeviceToHost );

    if(*equal_host)
        std::cout << "    Vectors are equal." << std::endl;
    else
        std::cout << "    Vectors are not equal." << std::endl;

    std::cout << "    Computation took " << time_span_gpu << " milliseconds." << std::endl << std::endl;

    std::chrono::time_point<std::chrono::system_clock> t1, t2;
    std::chrono::duration<double, std::milli>  time_span_cpu;

    std::cout << "2) Comparing vectors on cpu:" << std::endl;

    bool standard_equal = true;
    t1 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < LOOP; i++ )
    {
        int j = 0;
        while( standard_equal && j < N )
        {
            standard_equal = abs( x_h[j] - y_h[j] ) < eps;
            j++;
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    time_span_cpu = t2 - t1;
    
    if(standard_equal)
        std::cout << "  Vectors are equal." << std::endl;
    else
        std::cout << "  Vectors are not equal." << std::endl;

    std::cout << "  Computation on cpu took " << time_span_cpu.count() << " milliseconds." << std::endl << std::endl;

    //dealocating
    delete equal_host;
    delete[] x_h;
    delete[] y_h;
    cudaFree( equal_device_global );
    cudaFree( x_d );
    cudaFree( y_d );

    return 0;
}