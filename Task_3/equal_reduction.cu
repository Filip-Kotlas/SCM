//Filip Kotlas
//task 3 subtask 1

#include <iostream>
#include <chrono>


__global__ void equal( bool* are_equal, const float* const x, const float* const y, int N );
__global__ void equal( bool* are_equal, const float* const x, const float* const y, int N )
{
    extern __shared__ bool s_data[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_load = 1;
    const int stride = gridDim.x * blockDim.x * num_load;

    bool is_equal = true;
    for( unsigned int i = idx; i < N; i += stride )
    {
        if( x[i] != y[i] )
        {
            is_equal = false;
            break;
        }
    }
    s_data[threadIdx.x] = is_equal;
    __syncthreads();

    for( unsigned int str = blockDim.x/2; str > 0; str >>= 1 )
    {
        if (threadIdx.x < str)
            s_data[threadIdx.x] = s_data[threadIdx.x + str] ? s_data[threadIdx.x] : false;
        __syncthreads();
    }
    if( threadIdx.x == 0 ) 
        are_equal[blockIdx.x] = s_data[0];
}

__global__ void block_reduction(bool* data_arr, int size );
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
            s_data[tid] = s_data[tid + stride] ? s_data[tid] : false;
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

int main(void)
{
    float *x_h, *y_h; // vectors on host
    float *x_d, *y_d; // vectors on device

    const int N = 100000000;
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
    //Replace one component of vector to see wheater the code notices that the vectors don't match.
    x_h[N * 5 / 8] = 140;
    y_h[N * 5 / 8] = 140;

    //copying data from host to device
    cudaMemcpy(x_d, x_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, nBytes, cudaMemcpyHostToDevice);

    equal = are_equal(x_d, y_d, N );

    std::cout << "N: " << N << std::endl;

    std::cout << std::endl << "Comparing vectors on gpu:" << std::endl;

    cudaEventRecord(start);
    for( int i = 0; i < LOOP; i++ )
    {
        equal = are_equal(x_d, y_d, N );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_span_gpu, start, stop);

    if(equal)
        std::cout << "Vectors are equal." << std::endl;
    else
        std::cout << "Vectors are not equal." << std::endl;

    std::cout << "Computation on gpu took " << time_span_gpu << " milliseconds." << std::endl << std::endl;

    std::chrono::time_point<std::chrono::system_clock> t1, t2;
    std::chrono::duration<double, std::milli>  time_span_cpu;

    std::cout << "Comparing vectors on cpu:" << std::endl;

    bool standard_equal = true;
    t1 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < LOOP; i++ )
    {
        for( int j = 0; j < N; j++ )
        {
            if( x_h[j] != y_h[j] )
            {
                standard_equal = false;
                break;
            }
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    time_span_cpu = t2 - t1;
    
    if(standard_equal)
        std::cout << "Vectors are equal." << std::endl;
    else
        std::cout << "Vectors are not equal." << std::endl;

    std::cout << "Computation on cpu took " << time_span_cpu.count() << " milliseconds." << std::endl << std::endl;

    return 0;
}