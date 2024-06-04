#pragma once
#include <cassert>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

static cudaDataType         const REALTYPE = CUDA_R_64F;             //!< Data type for floating point in cuSPARSE


#define CHECK_CUDA(func)                                                           \
    {                                                                              \
        cudaError_t status = (func);                                               \
        if (status != cudaSuccess) {                                               \
            printf("CUDA API failed at line %d with error: %s (%d)\n",             \
                   __LINE__, cudaGetErrorString(status), status);                  \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    }
 
#define CHECK_CUSPARSE(func)                                                       \
    {                                                                              \
        cusparseStatus_t status = (func);                                          \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
                   __LINE__, cusparseGetErrorString(status), status);              \
            exit (EXIT_FAILURE);                                                   \
        }                                                                          \
    }

#define CHECK_CUBLAS(func)                                                         \
    {                                                                              \
        cublasStatus_t status = (func);                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                     \
            printf("CUBLAS API failed at line %d with error: %d\n",                \
                   __LINE__, status);                                              \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    }

#define CHECK_CUSOLVER(func)                                                       \
    {                                                                              \
        cusolverStatus_t status = (func);                                          \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
            printf("CUSOLVER API failed at line %d with error: %d\n",              \
                   __LINE__, status);                                              \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    }

#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }

/*
// See Dimitry Lyakh: CUDA Tutorial
//     https://github.com/DmitryLyakh/CUDA_Tutorial/blob/master/bla_lib.cu
//CUDA floating point data type selector:
template <typename T> struct CudaFPData{};
template <> struct CudaFPData<float>{
 using type = float;
 static constexpr cudaDataType_t kind = CUDA_R_32F;
};
template <> struct CudaFPData<double>{
 using type = double;
 static constexpr cudaDataType_t kind = CUDA_R_64F;
};
*/


// ############## kernel functions #####################################

/** @brief  Element-wise vector mult w_k = r_k*d_k.
 *
 * @param[in]  n  number of elements
 * @param[out] w  target vector
 * @param[in]  r  source vector
 * @param[in]  d  source vector
 *
 */
 
__global__ void vdmult_elem(int n, double* const w, double const* const r, double const* const d);


//**********************************************************************

/** @brief  Element-wise vector operation y_k = x_k + alpha*y_k.
 *
 * @param[in]      n  number of elements
 * @param[in]      x  source vector
 * @param[in]      alpha scalar
 * @param[in,out]  y  source/target vector
 *
 */
__global__ void vdxpay(int n, double const* const x, double alpha, double* const y);


// ############## other functions related to GPU #######################
//! Vector handler on GPU for cuSPARSE
class Vec {
public:
    Vec():m(0) {}

/** Allocates an uninitialized double vector on GPU.
 *  It is also provided for use in cuSPARSE.
 *
 * @param[in]  nelem  numer of elements
 */
    explicit
    Vec(size_t nelem): m(nelem)
    {
        size_t nBytes = m * sizeof(*ptr);
        CHECK_CUDA( cudaMalloc((void **) &ptr, nBytes) )  // cudaMallocManaged ??
        //CHECK_CUDA( cudaMallocManaged((void **) &ptr, nBytes) ) 
        CHECK_CUSPARSE( cusparseCreateDnVec(&vec, m, ptr, REALTYPE) )  // for MatVec
    }

/** Allocates a double vector on GPU ond copies the data from the 
 *  CPU vector @p v into it.
 *  It is also provided for use in cuSPARSE.
 *
 * @param[in]  v   vector on CPU
 */
    explicit
    Vec(std::vector<double> const &v): Vec(v.size())
    {
        CHECK_CUDA( cudaMemcpy(ptr, v.data(), m * sizeof(*ptr), cudaMemcpyHostToDevice) )
        //   cudaMemcpyAsync slows down my cg
        //CHECK_CUDA( cudaMemcpyAsync(ptr, v.data(), m * sizeof(*ptr), cudaMemcpyHostToDevice) )
    }

/** Dellocates the GPU memory for the vector 
 */
    ~Vec()
    {
        CHECK_CUSPARSE( cusparseDestroyDnVec(vec) )
        CHECK_CUDA( cudaFree(ptr) )
    }

    Vec(Vec const &)  = default;
    Vec(Vec &&)       = default;
    
/** Copies the GPU vector @p rhs to the GPU vector.
 *
 * @param[in] rhs   vector on GPU
 * @warning Equal number of elements are required.
 */    
    Vec& operator=(Vec const &rhs)
    {
        assert(rhs.m == m);
        cudaMemcpy(this->ptr, rhs.ptr, m * sizeof(*ptr), cudaMemcpyDeviceToDevice);
        return *this;
    }
    
/** Copies the CPU vector @p rhs to the GPU vector.
 *
 * @param[in] rhs   vector on CPU
 * @warning Equal number of elements are required.
 */    
    Vec &operator=(std::vector<double> const &rhs)
    {
        assert(rhs.size() == size());
        cudaMemcpy(this->ptr, rhs.data(), size() * sizeof(*ptr), cudaMemcpyHostToDevice);
        return *this;
    }
    Vec &operator=(Vec &&) = delete;

/**
 * @return number vector elements
*/
    size_t size() const { return m; };
/**
 * @return row pointer (GPU) to the starting address of vector
*/    
    double*       data()       { return this->ptr; };
/**
 * @return row pointer (GPU) to the starting address of vector
*/     
    double const* data() const { return this->ptr; };
/**
 * @return GPU vector handler for cuSPARSE
*/    
    auto const& sphandler() const {return this->vec;};
    
private:
    size_t				m;              //!<  number of elements
    double              *ptr = nullptr; //!<  address in device memory
    cusparseDnVecDescr_t vec = nullptr; //!<  handler for cuSPARSE
};
//----------------------------------------------------------------------

/** Vector scaling z_k := x_k*y_k on device
 *
 * @param[out] d_z resulting vector (preallocated)
 * @param[in]  d_x vector
 * @param[in]  d_y vector
 */
void vdmult_gpu(Vec &d_z, Vec const &d_x, Vec const &d_y);

/** Performs y_k := x_k +beta*y_k on device
 *
 * @param[in]     d_x vector
 * @param[in]     beta scalar
 * @param[in,out] d_y resulting vector 
 */
void vdxpay_gpu(Vec const &d_x, double beta, Vec &d_y);

/**
 * Copies device vector into an STL vector on host.
 *
 * @param[out]    h_v resulting STL vector (preallocated)
 * @param[in]     d_v host vector
 * 
 * @warning Identical data type and identical length of both vecors is assumed.
 */
void CopyDevice2Host(std::vector <double> &h_v, Vec const& d_v);



