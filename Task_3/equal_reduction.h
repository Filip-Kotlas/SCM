#include <iostream>
#include <chrono>

__global__ void equal( bool* are_equal, const float* const x, const float* const y, int N, float eps )
{
    extern __shared__ bool s_data[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_load = 1;
    const int stride = gridDim.x * blockDim.x * num_load;

    bool is_equal = true;
    unsigned int i = idx;
    while( is_equal && i < N )
    {
        is_equal = ( abs(x[i] - y[i]) < eps );  
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
bool are_equal(const float* const x_d, const float* const y_d, int N, float eps)
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

    equal <<< dimGrid, dimBlock, eq_Bytes >>>( eq_d, x_d, y_d, N, eps);

    const unsigned int oneBlock = 2 << (int)ceil(log(dimGrid.x + 0.0) / log(2.0));
    //std::cout << dimGrid.x << " " << oneBlock << " " << eq_Bytes << std::endl;
    block_reduction <<< 1, oneBlock, oneBlock *sizeof(bool) >>>(eq_d, dimGrid.x );

    bool equal;
    cudaMemcpy(&equal, eq_d, sizeof(bool), cudaMemcpyDeviceToHost);

    return equal;
}

__global__ void equal_with_one_common_variable( bool* are_equal_global, const float* const x, const float* const y, int N, float eps )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        if( ( abs(x[idx] - y[idx]) >= eps ) )
        {
            *are_equal_global = false;
        }
        // I have to use if to avoid race condition.
        //*are_equal_global = *are_equal_global && ( abs(x[idx] - y[idx]) < eps );
    }
}
