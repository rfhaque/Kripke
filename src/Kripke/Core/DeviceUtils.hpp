//
// Copyright (c) 2014-25, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the Kripke/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
  
#ifndef Kripke_DeviceUtils_HPP
#define Kripke_DeviceUtils_HPP

#if defined(KRIPKE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#elif defined(KRIPKE_ENABLE_HIP)
#include "hip/hip_runtime_api.h"
#endif

namespace Kripke {

void gpuMemcpyHtoD(void *dst, const void *src, size_t bytes) override
{
#if defined(KRIPKE_ENABLE_CUDA)
  return CuMemcpyHtoD(dst, src, bytes);
#elif defined(KRIPKE_ENABLE_HIP)
  return HipMemcpyHtoD(dst, src, bytes);
#endif
}

void gpuMemcpyDtoD(void* dst, const void* src, size_t bytes) override
{
#if defined(KRIPKE_ENABLE_CUDA)
  return CuMemcpyDtoD(dst, src, bytes);
#elif defined(KRIPKE_ENABLE_HIP)
  return HipMemcpyDtoD(dst, src, bytes);
#endif
}

void gpuMemcpyDtoH(void *dst, const void *src, size_t bytes) override
{
#if defined(KRIPKE_ENABLE_CUDA)
  return CuMemcpyDtoH(dst, src, bytes);
#elif defined(KRIPKE_ENABLE_HIP)
  return HipMemcpyDtoH(dst, src, bytes);
#endif
}

#if defined(KRIPKE_ENABLE_CUDA)
inline void gpuErrorCheck(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr, "[KRIPKE] GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}
#elif defined(KRIPKE_ENABLE_HIP)
inline void gpuErrorCheck(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) {
      fprintf(stderr, "[KRIPKE] GPU Error: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}
#endif

#define KRIPKE_GPU_ERROR_CHECK(code) { gpuErrorCheck((code), __FILE__, __LINE__); }

inline void gpuDeviceSynchronize() {
#if defined(KRIPKE_ENABLE_CUDA)
   KRIPKE_GPU_ERROR_CHECK(cudaDeviceSynchronize());
#elif defined(KRIPKE_ENABLE_HIP)
   KRIPKE_GPU_ERROR_CHECK(hipDeviceSynchronize());
#endif
}

#endif // Kripke_DeviceUtils_HPP

}  // end of namespace Kripke
