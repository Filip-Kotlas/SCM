#pragma once

#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <unordered_map>

//#####################################
// G.Haase
// See https://sourceforge.net/p/predef/wiki/Compilers/
//     http://www.cplusplus.com/doc/tutorial/preprocessor/
//  also:  export OMP_DISPLAY_ENV=VERBOSE
//#####################################
/** 	Checks for compilers, its versions, threads etc. on CPU
 * 
	@param[in] argc	number of command line arguemnts
	@param[in] argv	command line arguments as array of C-strings
*/
template <class T>
void check_env(T argc, char const *argv[])
{
    std::cout << "\n#######################################################################\n";
    std::cout << "Code    :";
    for (T k = 0; k < argc; ++k) std::cout << "  " << argv[k];
    std::cout << std::endl;

    // compiler:      https://sourceforge.net/p/predef/wiki/Compilers/
    std::cout <<    "Compiler:  ";
#if defined __INTEL_COMPILER
#pragma message(" ##########  INTEL  ###############")
    std::cout << "INTEL " << __INTEL_COMPILER;
    // Ignore warnings for #pragma acc   unrecognice
#pragma warning disable 161
    // Ignore warnings for #pragma omp   unrecognice
#pragma warning disable 3180

#elif defined __NVCC__
//#pragma message(" ##########  NVCC    ###############")
//  https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
    std::cout << "NVCC " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ ;
#elif defined  __clang__
#pragma message(" ##########  CLANG    ###############")
    std::cout << "CLANG " << __clang_major__ << "." << __clang_minor__ << "."; // << __clang_patchlevel__;
#elif defined __GNUC__
//#pragma message(" ##########  Gnu    ###############")
    std::cout << "Gnu " <<  __GNUC__  << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#else
#pragma message(" ##########  unknown Compiler   ###############")
    std::cout << "unknown";
#endif
    std::cout << "  C++ standard: " << __cplusplus << std::endl;
    
    // Parallel environments
    std::cout <<    "Parallel:  ";
#if defined MPI_VERSION
#ifndef __GNUC__
#pragma message(" ##########  MPI    ###############")
#endif
#ifdef OPEN_MPI
    std::cout << "OpenMPI ";
#else
    std::cout << "MPI ";
#endif
    std::cout << MPI_VERSION << "." << MPI_SUBVERSION << "   ";
#endif

#ifdef _OPENMP
//https://www.openmp.org/specifications/
//https://stackoverflow.com/questions/1304363/how-to-check-the-version-of-openmp-on-linux
    std::unordered_map<unsigned, std::string> const map{
        {200505, "2.5"}, {200805, "3.0"}, {201107, "3.1"}, {201307, "4.0"}, {201511, "4.5"}, {201611, "5.0"}, {201811, "5.0"}};
#ifndef __GNUC__
#pragma message(" ##########  OPENMP    ###############")
#endif
    std::cout << "OpenMP ";
    try {
        std::cout << map.at(_OPENMP);
    }
    catch (...) {
        std::cout << _OPENMP;
    }
    #pragma omp parallel
    {
        #pragma omp master
        {
            const int nn = omp_get_num_threads();          // OpenMP
            std::cout << " ---> " <<  nn << " Threads   ";
        }
        #pragma omp barrier
    }

#endif
#ifdef _OPENACC
#pragma message(" ##########  OPENACC    ###############")
    std::cout << "OpenACC   ";
#endif
    std::cout << std::endl;
    std::cout << "Date    :  " << __DATE__ << "  " << __TIME__;
    std::cout << "\n#######################################################################\n";
}



/** @brief  Lists basic properties of GPU
 */
inline
void printGPUInfo()
{
    using std::cout, std::endl, std::boolalpha;

    cout <<"\n++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    // https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
    int nDevices;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) printf("%i : %s\n", err, cudaGetErrorString(err));
    
    //  https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html 
    cudaDeviceProp prop;
    // https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html
    err = cudaGetDeviceProperties (&prop, 0);
    if (err != cudaSuccess) printf("%i : %s\n", err, cudaGetErrorString(err));
    cout << "We work on " << nDevices << " x " << prop.name << " GPU \n    with "
         << prop.multiProcessorCount << " Multiprocessors (SME)"
         << ", Compute capability " << prop.major << "." << prop.minor << endl;
    cout << "global Mem: " << prop.totalGlobalMem/1024/1000/1000 << " GB" << endl;
    cout << "shared Mem per SME: " << prop.sharedMemPerMultiprocessor/1024 << "kB"
         << "   shared Mem per Block: " << prop.sharedMemPerBlock/1024 << " kB"
         << "   32b-Registers per Block: " << prop.regsPerBlock << endl;
    cout << "global L2 cache: " << prop.l2CacheSize/1024 << " kB"
         << "   local L1 cache supported: " << boolalpha << bool(prop.localL1CacheSupported)  << endl;
    cout << "max Threads per Block:" << prop.maxThreadsPerBlock << endl;
    cout <<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
}


