#ifndef FILE_MATRIX
#define FILE_MATRIX

#ifdef __INTEL_CLANG_COMPILER
#pragma message(" ##########  Use of MKL  ###############")
#include <mkl.h>
//#include <mkl_boost_ublas_matrix_prod.hpp>
// MKL
#else
#pragma message(" ##########  Use of CBLAS  ###############")
//extern "C"
//{
#include <openblas/cblas.h>               // cBLAS Library
//#include <openblas/lapacke.h>             // Lapack
//}
#endif

#include <cassert>
#include <chrono>
#include <cmath>           // exp()
#include <cublas_v2.h>    //  new CUDA API, see https://docs.nvidia.com/cuda/cublas/index.html#new-and-legacy-cublas-api
//#include </opt/cuda/targets/x86_64-linux/include/cublas_v2.h>
//#include <tensorflow/third_party/gpus/cuda/include/cublas_v2.h>    //  new CUDA API, see https://docs.nvidia.com/cuda/cublas/index.html#new-and-legacy-cublas-api
#include <cuda_runtime.h> // cudaMalloc, cudaFree,cudaGetDeviceProperties
#include <iostream>
#include <limits>
#include <vector>


/** 	Dense matrix, columnwise stored as 1D vector.
 */
template <class T>
class denseMatrix
{
public:
    /** 	Constructor with implicit initialization
     *
     * 	@param[in] nrow  number of rows
     * 	@param[in] mcol  number of columns
     *
     */
    denseMatrix(int nrow, int mcol);

    /** 	Constructor
     *
     * 	@param[in] nrow  number of rows
     * 	@param[in] mcol  number of columns
     *  @param[in] val   initial value of matrix elements
     *
     */
    denseMatrix(int nrow, int mcol, T val);

    denseMatrix(denseMatrix<T> const &) = default;
    denseMatrix(denseMatrix<T> &&) = default;
    denseMatrix<T> &operator=(denseMatrix<T> const &) = default;
    denseMatrix<T> &operator=(denseMatrix<T> &&) = default;

    /** 	Matrix-matrix multiplication (columnwise) access.
     *
     * 	@param[in] B  matrix
     * 	@return resulting matrix @f$ A*B @f$
     *
     */
    denseMatrix<T>  Mult(denseMatrix<T> const &B) const;


    /** 	Matrix-matrix multiplication (columnwise) access.
     *
     * 	@param[in] B  matrix
     * 	@return resulting matrix @f$ A*B @f$
     *
     */
    denseMatrix<T>  Mult_fast(denseMatrix<T>  const &B) const;
    denseMatrix<T>  Mult_OpenMP(denseMatrix<T> const &B) const;

    /** 	Matrix-matrix multiplication (columnwise) access.
     *
     * 	@param[in] B  matrix
     * 	@return resulting matrix @f$ A*B @f$
     *
     */
    denseMatrix<T>  Mult_Blas(denseMatrix<T>  const &B) const;
    denseMatrix<T>  Mult_cuBlas(denseMatrix<T>  const &B) const;
    //denseMatrix Mult_cuBlas_row(denseMatrix const &B) const;  // doesn't work

    denseMatrix<T>  Mult_cuBlas_Test(denseMatrix<T>  const &B, int loopcount, double &seconds) const;
    denseMatrix<T>  Mult_cuOpenMP_Test(denseMatrix<T>  const &B, int loopcount, double &seconds) const;


    /** 	Compares two matrices for equality.
     *
     * 	@param[in] B  matrix
     * 	@return equal regarding internal relative accuracy?
     *
     */
    bool operator==(denseMatrix<T>  const &B) const;

    /** 	Get the number of rows.
    * 	@return number of rows.
    */
    int GetNrows() const
    {
        return _nrows;
    }
    /** 	Get the number of rows.
    * 	@return number of rows.
    */

    int GetNcols() const
    {
        return _mcols;
    }

    /** 	Access to element @p i in columnwise numbering
    * 	@return value.
    */
    T &operator[] (int i)
    {
        return _A[i];
    }

    /** 	Access to element @p i in columnwise numbering
    * 	@return value.
    */
    T const &operator[] (int i) const
    {
        return _A[i];
    }

    /**
     *  @return raw pointer to begin of element data.
    */
    T *data()
    {
        return _A.data();
    }

    /**
     *  @return raw pointer to begin of element data.
    */
    T const *data() const
    {
        return _A.data();
    }

private:
    int _nrows;       //!< number of rows in matrix
    int _mcols;       //!< number of coluns in matrix

protected:
    std::vector<T> _A;  //!< matrix values

};

/** 	Operator overloading: Matrix-vector product with the multiplication operator
 * 	@param[in] A	dense matrix (1D access)
 *  @param[in] u	vector
 *
 *	@return    resulting vector
*/#
template <class T>
inline
std::vector<T> operator*(denseMatrix<T>  const &A, std::vector<T> const &u)
{
    return A.Mult(u);
}
//----------------------------------------------------------------------


//----------------------------------------------------------------------

///** Output operator for vector
//* 	@param[in,out] s	output stream, e.g. @p cout
//*  @param[in]     v    vector
//*
//*	@return    output stream
//*/
//template <class T>
//std::ostream &operator<<(std::ostream &s, std::vector<T> const &v)
//{
//for (auto vp : v) {
//s << vp << "  ";
//}
//return s;
//}

/** Generates an equistant vector having @p n elements from interval [@p a, @p b].
 *
 *  @param[in] a    interval start
 *  @param[in] b    interval end
 *  @param[in] n    number of equidistant points in interval
 *
 *  @return    vector with the equistant elements
*/
template <class T>
std::vector<T> linspace(T a, T b, int n = 100)
{
    std::vector<T> x(n);
    T const h = (b - a) / T(n - 1);
    for (size_t k = 0; k < x.size(); ++k)
    {
        x[k] = T(k) * h + a;
    }
    return x;
}


/** <a href="https://en.wikipedia.org/wiki/Sigmoid_function">Sigmoid</a> function.
 *
 *  @param[in] x  evaluation point
 *
 *  @return    sigmoid value
*/
template <class T>
T sigmoid(T x)
{
    return T(1.0) / (T(1.0) + std::exp(x));
}

template <class T>
denseMatrix<T>::denseMatrix(int const nrow, int const mcol)
    : _nrows(nrow), _mcols(mcol), _A(_nrows * _mcols)
{
    int const nm = std::max(_nrows, _mcols);
    auto const x = linspace(T(-5), T(5), nm);

    std::vector<T> fx(nm);
    for (size_t k = 0; k < fx.size(); ++k)
    {
        fx[k] = sigmoid(-x[k]);
    }

#pragma omp parallel for
    for (int i = 0; i < _nrows; ++i)
    {
        for (int j = 0; j < _mcols; ++j)
        {
            _A[i * _mcols + j] = fx[i] * fx[j];
        }
    }

}

template <class T>
denseMatrix<T>::denseMatrix(int const nrow, int const mcol, T const val)
    : _nrows(nrow), _mcols(mcol), _A(_nrows * _mcols, val)
{
}

template <class T>
bool denseMatrix<T>::operator==(denseMatrix<T> const &B) const
{
    denseMatrix<T> const  &A = *this;
    assert(A.GetNrows() == B.GetNrows());
    assert(A.GetNcols() == B.GetNcols());

    T const eps = std::sqrt(std::numeric_limits<T>::epsilon()) ;
    bool bsame = true;
    for (int k = 0; k < A.GetNrows()*A.GetNcols(); ++k)
    {
        bool bk = std::abs(A[k] - B[k]) <= eps * (1 + T(0.5) * ( std::abs(A[k]) + std::abs(B[k]) ));
        if (!bk)
        {
            std::cerr << "matrices differ at [" << k / A.GetNcols() << "," << k % A.GetNcols() << "]";
            std::cerr << " :  " << A[k] << "  vs. " << B[k] << std::endl;
        }
        bsame &= bk;
    }
    return bsame;
}

template <class T>
denseMatrix<T> denseMatrix<T>::Mult(denseMatrix<T> const &B) const
{
    denseMatrix<T> const  &A = *this;
    assert( A.GetNcols() == B.GetNrows() ); // inner dimensions equal?

    denseMatrix<T> C(A.GetNrows(), B.GetNcols());
    int const acols = A.GetNcols();
    int const arows = A.GetNrows();
    //int const bcols = B.GetNcols();
    int const brows = B.GetNrows();
    int const ccols = C.GetNcols();
    int const crows = C.GetNrows();

    for (int i = 0; i < crows; ++i)
    {
        for (int j = 0; j < ccols; ++j)
        {
            T tmp = 0.0;
            for (int k = 0; k < acols; ++k)
            {
                tmp += A[i + k * arows] * B[k + j * brows];
            }
            C[i + j * crows] = tmp;
        }
    }

    return C;
}

template <class T>
denseMatrix<T> denseMatrix<T>::Mult_fast(denseMatrix<T> const &B) const
{
    denseMatrix<T> const  &A = *this;
    assert( A.GetNcols() == B.GetNrows() ); // inner dimensions equal?

    denseMatrix<T> C(A.GetNrows(), B.GetNcols(), 0.0);
    int const acols = A.GetNcols();
    int const arows = A.GetNrows();
    //int const bcols = B.GetNcols();
    int const brows = B.GetNrows();
    int const ccols = C.GetNcols();
    int const crows = C.GetNrows();

    for (int j = 0; j < ccols; ++j)
    {
        for (int k = 0; k < acols; ++k)
        {
            for (int i = 0; i < crows; ++i)
            {
                C[i + j * crows] += A[i + k * arows] * B[k + j * brows];
            }
        }
    }

    return C;
}

template <class T>
denseMatrix<T> denseMatrix<T>::Mult_OpenMP(denseMatrix<T> const &B) const
{
    denseMatrix<T> const  &A = *this;
    assert( A.GetNcols() == B.GetNrows() ); // inner dimensions equal?

    denseMatrix<T> C(A.GetNrows(), B.GetNcols(), 0.0);
    int const acols = A.GetNcols();
    int const arows = A.GetNrows();
    //int const bcols = B.GetNcols();
    int const brows = B.GetNrows();
    int const ccols = C.GetNcols();
    int const crows = C.GetNrows();

    #pragma omp parallel for //collapse(2)
    for (int j = 0; j < ccols; ++j)
    {
        for (int k = 0; k < acols; ++k)
        {
            for (int i = 0; i < crows; ++i)
            {
                C[i + j * crows] += A[i + k * arows] * B[k + j * brows];
            }
        }
    }

    return C;
}

template <class T>
denseMatrix<T>  denseMatrix<T>::Mult_cuOpenMP_Test(denseMatrix<T>  const &B, int loopcount, double &seconds) const
{
    using namespace std;
    using namespace std::chrono;    // timing
    denseMatrix<T> const  &A = *this;
    assert( A.GetNcols() == B.GetNrows() ); // inner dimensions equal?

    denseMatrix<T> C(A.GetNrows(), B.GetNcols(), 0.0);         // fast
    int const acols = A.GetNcols();
    int const arows = A.GetNrows();
    //int const bcols = B.GetNcols();
    int const brows = B.GetNrows();
    int const ccols = C.GetNcols();
    int const crows = C.GetNrows();


    //#pragma omp targetmap( to : v1 [ 0 :N] , v2 [ 0 :N] , p [ 0 :N] )
    #pragma omp target map (to: A, B) map(tofrom: C)
    {
        auto t1 = system_clock::now();
        for (int loop = 0; loop < loopcount; ++loop)
        {
            for (int j = 0; j < ccols; ++j)
            {
                for (int k = 0; k < acols; ++k)
                {
                    for (int i = 0; i < crows; ++i)
                    {
                        C[i + j * crows] += A[i + k * arows] * B[k + j * brows];
                    }
                }
            }
        }
        auto t2 = system_clock::now();
        auto scd = duration_cast<microseconds>(t2 - t1) / loopcount; //1e6/
        seconds = std::chrono::duration<double>(scd).count();
    }

    return C;
}


//  from cblas.h
//  typedef CBLAS_LAYOUT CBLAS_ORDER; /* this for backward compatibility with CBLAS_ORDER */
//-------
template<class T>
//void gemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const T alpha, const T *A, const int lda, const T *B, const int ldb, const T beta, T *C, const int ldc)
//void gemm(const CBLAS_ORDER layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const T alpha, const T *A, const int lda, const T *B, const int ldb, const T beta, T *C, const int ldc)
void gemm(const CBLAS_ORDER, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE, const int, const int, const int, const T, const T *, const int, const T *, const int, const T, T *, const int )
{
//  https://stackoverflow.com/questions/13636540/how-to-check-for-the-type-of-a-template-parameter
    //if constexpr(std::is_same<T,double>)
    //{
    //std::cout << "DGEMM ";
    //cblas_dgemm(layout, TransA, TransB,
    //M, N, K,
    //alpha, A, lda,
    //B, ldb,
    //beta, C, ldc);
    //}
    //else if constexpr(std::is_same<T,float>)
    //{
    //std::cout << "SGEMM ";
    //cblas_sgemm(layout, TransA, TransB,
    //M, N, K,
    //alpha, A, lda,
    //B, ldb,
    //beta, C, ldc);
    //}
    //else
    //{
    //static_assert(0,"No appropriate type.");
    //}
    assert(false);
}

template<> inline
//void gemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
void gemm(const CBLAS_ORDER layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    //std::cout << "DGEMM ";
    cblas_dgemm(layout, TransA, TransB,
                M, N, K,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
}

template<> inline
//void gemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
void gemm(const CBLAS_ORDER layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    //std::cout << "SGEMM ";
    cblas_sgemm(layout, TransA, TransB,
                M, N, K,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
}

//-------

template <class T>
denseMatrix<T> denseMatrix<T>::Mult_Blas(denseMatrix<T> const &B) const
{
//  C_(MxN) := A_(MxK) * B(KxN)
//  rowise storage in all three matrices

    denseMatrix<T> const  &A = *this;
    assert( A.GetNcols() == B.GetNrows() ); // inner dimensions equal?
    denseMatrix<T> C(A.GetNrows(), B.GetNcols(), 0.0);

    T const *const pA = A.data();
    T const *const pB = B.data();
    T       *const pC = C.data();

    T const alpha = T(1.0);
    T const beta = T(0.0);
//      https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
//      http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
//      http://www.netlib.org/lapack/explore-html/dc/d18/cblas__dgemm_8c.html
    unsigned int const M = C.GetNrows();
    unsigned int const N = C.GetNcols();
    unsigned int const K = A.GetNcols();
    unsigned int const LDA = M;
    unsigned int const LDB = K;
    unsigned int const LDC = M;

    gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
         M, N, K,
         alpha, pA, LDA,
         pB, LDB,
         beta, pC, LDC);
    return C;
}

//-------
template<class T>
cublasStatus_t cublasgemm(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const T          *alpha,
                          const T          *A, int lda,
                          const T          *B, int ldb,
                          const T          *beta,
                          T          *C, int ldc)
{
    assert(false);
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<> inline
cublasStatus_t cublasgemm(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const double          *alpha,
                          const double          *A, int lda,
                          const double          *B, int ldb,
                          const double          *beta,
                          double          *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<> inline
cublasStatus_t cublasgemm(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const float          *alpha,
                          const float          *A, int lda,
                          const float          *B, int ldb,
                          const float          *beta,
                          float          *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

//-------


template <class T>
denseMatrix<T> denseMatrix<T>::Mult_cuBlas_Test(denseMatrix<T> const &B, int const loopcount, double &seconds) const
{
    using namespace std;
    using namespace std::chrono;    // timing
    denseMatrix<T> const  &A = *this;
    assert( A.GetNcols() == B.GetNrows() ); // inner dimensions equal?
    denseMatrix<T> C(A.GetNrows(), B.GetNcols(), 0.0);

    T const *const pA_h = A.data();    // host data
    T const *const pB_h = B.data();
    T       *const pC_h = C.data();

    T const alpha = T(1.0);
    T const beta = T(0.0);
//      https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
//      http://www.netlib.org/lapack/explore-html/dc/d18/cblas__dgemm_8c.html
    unsigned int const M = C.GetNrows();
    unsigned int const N = C.GetNcols();
    unsigned int const K = A.GetNcols();
    //unsigned int const LDA = K;
    //unsigned int const LDB = N;
    //unsigned int const LDC = N;

//  -- Initialize cuBLAS, check GPU
//  https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
    cublasStatus_t  stat;                //  CUBLAS  functions  status
    cublasHandle_t  handle;

    stat = cublasCreate (& handle );   //  initialize  CUBLAS  context
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cerr << "ERROR: cublasInit() failed!\n";
        exit(1);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties (&prop, 0);
    std::cout << std::endl << "We work on a " << prop.name << " GPU  with " <<
              prop.multiProcessorCount << " Multiprocessors" <<  std::endl;

//  -- Move data from host to device
// allocate memory on device
    cudaError_t  cudaStat;               //  cudaMalloc  status
    T *A_d, *B_d, *C_d; // device data
    cudaStat = cudaMalloc(reinterpret_cast<void **>(&A_d), M * K * sizeof(*A_d));
    cudaStat = cudaMalloc(reinterpret_cast<void **>(&B_d), K * N * sizeof(*B_d));
    cudaStat = cudaMalloc(reinterpret_cast<void **>(&C_d), M * N * sizeof(*C_d));
    if (cudaStat != cudaSuccess)
    {
        cerr << cudaGetErrorName (cudaStat ) << endl;
        exit(1);
    }

//  copy data:  host --> device
    stat = cublasSetMatrix(M, K, sizeof(*A_d), pA_h, M, A_d, M);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cerr << "A: cublasSetMatrix ERROR: " << stat << endl;
        exit(1);
    }
    stat = cublasSetMatrix(K, N, sizeof(*B_d), pB_h, K, B_d, K);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cerr << "B: cublasSetMatrix ERROR: " << stat << endl;
        exit(1);
    }
    stat = cublasSetMatrix(M, N, sizeof(*C_d), pC_h, M, C_d, M);    // C^T  on device
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cerr << "C: cublasSetMatrix ERROR: " << stat << endl;
        exit(1);
    }

    auto t1 = system_clock::now();
    for (int k = 0; k < loopcount; ++k)
    {
//  arithm.
//  CUBLAS_OP_N , see https://docs.nvidia.com/cuda/cublas/index.html#cublasoperation_t
        // C = A*B   // only correct if we use also columnwise storage
        cudaDeviceSynchronize();
        stat = cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A_d, M, B_d, K, &beta, C_d, M);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            cerr << "cublasDgemm ERROR: " << stat << endl;
            exit(1);
        }
    }
    cudaDeviceSynchronize();
    auto t2 = system_clock::now();
    auto scd = duration_cast<microseconds>(t2 - t1) / loopcount; //1e6/
    //  https://stackoverflow.com/questions/57538507/how-to-convert-stdchronoduration-to-double-seconds
    seconds = std::chrono::duration<double>(scd).count();

//  copy data: device --> host
    stat = cublasGetMatrix(M, N, sizeof(*C_d), C_d, M, pC_h, M);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cerr << "C: cublasSetMatrix ERROR: " << stat << endl;
        exit(1);
    }

//  -- Free all cuBlas ressources
    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);
    cublasDestroy(handle);

    return C;
}



//----------------------------------------------------
template <class T>
//constexpr                  // icpc: string no enumeration type for constexpr function
std::string GetPrecision()
{
    if (std::is_same<T, double>::value)
        return "DP";
    else if (std::is_same<T, float>::value)
        return "SP";
    else
        return "XX";
}

template <class T>
void Performance_Test()
{
    using namespace std;
    using namespace std::chrono;    // timing

    string precision = GetPrecision<T>();
    cout << "\n#########################################################\n";
    //cout << " Run time for DenseMatrix GEMMult: " << endl;

    //int const M = 3;
    //int const N = 3;
    //int const K = 3;
    //int const M = 200;
    //int const N = 500;
    //int const K = 300;
    int const M = 2000;
    int const N = 2500;
    int const K = 3000;
    denseMatrix<T> const A(M, K), B(K, N);
    denseMatrix<T> D(M, N);
    
    cout << M << " x " << K << "  mult  " << K << " x " << N << endl << endl; 

    //{
    //auto t1 = system_clock::now();
    //denseMatrix<T> C = A.Mult(B);
    //auto t2 = system_clock::now();
    //auto tdiff = duration_cast<microseconds>(t2 - t1);
    //auto seconds = std::chrono::duration<double>(tdiff).count();
    //cout << " Mult_OpenMP: " << seconds << " sec." ;
    //cout << " (" << 2.0 * M *N *K / 1e9 / seconds << " GFLOPS(" << precision << "))\n";
    //D=C;
    //}

    {
        auto t1 = system_clock::now();
        denseMatrix<T> C = A.Mult_fast(B);
        auto t2 = system_clock::now();
        auto tdiff = duration_cast<microseconds>(t2 - t1);
        auto seconds = std::chrono::duration<double>(tdiff).count();
        cout << " Mult_sequ  : " << seconds << " sec." ;
        cout << " (" << 2.0 * M *N *K / 1e9 / seconds << " GFLOPS(" << precision << "))\n";
        //assert(D==C);
        D = C;
    }

    {
        auto t1 = system_clock::now();
        denseMatrix<T> C = A.Mult_OpenMP(B);
        auto t2 = system_clock::now();
        auto tdiff = duration_cast<microseconds>(t2 - t1);
        auto seconds = std::chrono::duration<double>(tdiff).count();
        cout << " Mult_OpenMP: " << seconds << " sec." ;
        cout << " (" << 2.0 * M *N *K / 1e9 / seconds << " GFLOPS(" << precision << "))\n";
        assert(D == C);
        //D=C;
    }

    {
        auto t1 = system_clock::now();
        denseMatrix<T> C = A.Mult_Blas(B);
        auto t2 = system_clock::now();
        auto tdiff = duration_cast<microseconds>(t2 - t1);
        auto seconds = std::chrono::duration<double>(tdiff).count();
        cout << " Mult_BLAS  : " << seconds << " sec." ;
        cout << " (" << 2.0 * M *N *K / 1e9 / seconds << " GFLOPS(" << precision << "))\n";
        assert(D == C);
        //D=C;
    }

    {
        cout << "\n#########################################################\n";
        cout << "\n################   GPU   ################################\n";
        cout << " Run time for DenseMatrix cuGEMMult: " << endl;

        double seconds{ -1.0};
        denseMatrix<T> C = A.Mult_cuBlas_Test(B, 20, seconds);
        cout << " Mult_cuBlas: " << seconds << " sec." ;
        cout << " (" << 2.0 * M *N *K / 1e9 / seconds << " GFLOPS(" << precision << "))\n";
        assert(D == C);
    }

    //{
    //cout << "\n################   GPU OpenMP   ################################\n";
    //cout << " Run time for DenseMatrix Mult: " << endl;

    //double seconds{ -1.0};
    //denseMatrix<T> C = A.Mult_cuOpenMP_Test(B, 1, seconds);
    //cout << " Mult_cuBlas: " << seconds << " sec." ;
    //cout << " (" << 2.0 * M *N *K / 1e9 / seconds << " GFLOPS(" << precision << "))\n";
    ////cout << " (" << 2.0 * M *N *K / pow(1024.0, 3.0) / seconds << " GFLOPS(" << precision << "))\n";
    //assert(D == C);
    //}
}


template <class T>
void Performance_Test_Large()
{
    using namespace std;
    using namespace std::chrono;    // timing

    string precision = GetPrecision<T>();
    cout << "\n#############  L A R G E   ######################\n";
    //cout << " Run time for DenseMatrix GEMMult: " << endl;

    int const M = 20000/2*3;               // 16 GB for one matrix (double) needed
    int const N = 25000/2*3;
    int const K = 30000/2*3;
    denseMatrix<T> const A(M, K), B(K, N);
    denseMatrix<T> D(M, N);
    
    cout << M << " x " << K << "  mult  " << K << " x " << N << endl << endl; 

    {
        auto t1 = system_clock::now();
        denseMatrix<T> C = A.Mult_Blas(B);
        auto t2 = system_clock::now();
        auto tdiff = duration_cast<microseconds>(t2 - t1);
        auto seconds = std::chrono::duration<double>(tdiff).count();
        cout << " Mult_BLAS  : " << seconds << " sec." ;
        cout << " (" << 2.0 * M *N *K / 1e9 / seconds << " GFLOPS(" << precision << "))";
        //assert(D == C);
        D=C;
    }

    {
//        cout << "\n#########################################################\n";
        cout << "\n################   GPU   ################################\n";
        cout << " Run time for DenseMatrix cuGEMMult: " << endl;

        double seconds{ -1.0};
        denseMatrix<T> C = A.Mult_cuBlas_Test(B, 20, seconds);
        cout << " Mult_cuBlas: " << seconds << " sec." ;
        cout << " (" << 2.0 * M *N *K / 1e9 / seconds << " GFLOPS(" << precision << "))\n";
        assert(D == C);
    }

}

//#####################################################################




#endif
