#ifndef _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CURAND_HPP
#define _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CURAND_HPP
#endif

#if CUDAPP_CUDA_VERSION >= 3020
  #include <curand.h>

  #ifdef CUDAPP_TRACE_CUDA
    #define CURAND_PRINT_ERROR_TRACE(NAME, CODE) \
      if (CODE != CURAND_STATUS_SUCCESS) \
        std::cerr << NAME << " failed with code " << CODE << std::endl;
  #else
    #define CURAND_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
  #endif

  #define CURAND_CALL_GUARDED(NAME, ARGLIST) \
    { \
      CUDAPP_PRINT_CALL_TRACE(#NAME); \
      curandStatus_t cu_status_code; \
      cu_status_code = NAME ARGLIST; \
      CURAND_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
      if (cu_status_code != CURAND_STATUS_SUCCESS) \
        throw pycuda::error(#NAME, CUDA_SUCCESS);\
    }
#else
  #define CURAND_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
  #define CURAND_CALL_GUARDED(NAME, ARGLIST) /*nothing*/
#endif



