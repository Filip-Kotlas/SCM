//Filip Kotlas
//task 3 subtask 1

#include <iostream>
#include <chrono>

#define EPS 1e-4

__global__ void equal( bool* are_equal, const float* const x, const float* const y, int N )
{
    extern __shared__ bool s_data[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_load = 1;
    const int stride = gridDim.x * blockDim.x * num_load;

    bool is_equal = true;
    unsigned int i = idx;
    while( is_equal && i < N )
    {
        is_equal = ( abs(x[i] - y[i]) < EPS );  
        i += stride;
    }
    s_data[threadIdx.x] = is_equal;
    __syncthreads();

    for( unsigned int str = blockDim.x/2; str > 0; str >>= 1 )
    {
        if (threadIdx.x < str)
            s_data[threadIdx.x] = s_data[threadIdx.x + str] && s_data[threadIdx.x];
            //s_data[threadIdx.x] = s_data[threadIdx.x + str] ? s_data[threadIdx.x] : false;
        __syncthreads();
    }
    if( threadIdx.x == 0 ) 
        are_equal[blockIdx.x] = s_data[0];
}

__global__ void block_reduction(bool* data_arr, int size )
{
    if( size > static_cast<int>(blockDim.x) )
        return;

    extern __shared__ bool s_data[];
    const int tid = threadIdx.x;
    if( tid < size )
        s_data[tid] = data_arr[tid];
    else
        s_data[tid] = true;

    __syncthreads();
    for( int stride = blockDim.x / 2; stride > 0; stride >>= 1 )
    {
        if( tid < stride )
        {
            s_data[tid] = s_data[tid + stride] && s_data[tid];
            //s_data[tid] = s_data[tid + stride] ? s_data[tid] : false;
        }
        __syncthreads();
    }
    if (tid == 0)
        data_arr[0] = s_data[0];
}

static bool b_init = false;
//host function
bool are_equal(const float* const x_d, const float* const y_d, int N)
{
    static bool *eq_d;
    static dim3 dimBlock, dimGrid;
    static int eq_Bytes;
    if (!b_init)   // only  in the first function call
    {
        b_init = !b_init;
        cudaDeviceProp prop;
        cudaGetDeviceProperties (&prop, 0);

        const int blocksize = 8 * 64,
                  gridsize = prop.multiProcessorCount;
        eq_Bytes   = blocksize * sizeof(bool);
        dimBlock = dim3(blocksize);
        dimGrid  = dim3(1 * gridsize);

        cudaMalloc((void **) &eq_d, blocksize * sizeof(bool)); // temp. memory on device
    }

    equal <<< dimGrid, dimBlock, eq_Bytes >>>( eq_d, x_d, y_d, N);

    const unsigned int oneBlock = 2 << (int)ceil(log(dimGrid.x + 0.0) / log(2.0));
    //std::cout << dimGrid.x << " " << oneBlock << " " << eq_Bytes << std::endl;
    block_reduction <<< 1, oneBlock, oneBlock *sizeof(bool) >>>(eq_d, dimGrid.x );

    bool equal;
    cudaMemcpy(&equal, eq_d, sizeof(bool), cudaMemcpyDeviceToHost);

    return equal;
}

__global__ void equal_with_one_common_variable( bool* are_equal_global, const float* const x, const float* const y, int N )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        if( ( abs(x[idx] - y[idx]) >= EPS ) )
        {
            *are_equal_global = false;
        }
        // I have to use if to avoid race condition.
        //*are_equal_global = *are_equal_global && ( abs(x[idx] - y[idx]) < EPS );
    }
}

int main(void)
{
    float *x_h, *y_h; // vectors on host
    float *x_d, *y_d; // vectors on device

    const long int N = 100000000;
    const int nBytes = N * sizeof(float);
    const int LOOP = 10;
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
    x_h[index + index] = 145;
    y_h[index + index] = 140;
    //y_h[N-1] = 145;

    //copying data from host to device
    cudaMemcpy(x_d, x_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, nBytes, cudaMemcpyHostToDevice);

    equal = are_equal(x_d, y_d, N );

    std::cout << "N: " << N << std::endl;
    std::cout << "LOOP: " << LOOP << std::endl;

    std::cout << std::endl << "1) Comparing vectors on gpu:" << std::endl;
    std::cout << "  a) Standard algorithm: " << std::endl;

    cudaEventRecord(start);
    for( int i = 0; i < LOOP; i++ )
    {
        equal = are_equal(x_d, y_d, N );
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
        equal_with_one_common_variable<<< dimGrid, dimBlock >>>(equal_device_global, x_d, y_d, N);
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
            standard_equal = abs( x_h[j] - y_h[j] ) < EPS;
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