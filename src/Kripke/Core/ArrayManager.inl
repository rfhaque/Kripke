//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ArrayManager_INL
#define CHAI_ArrayManager_INL

#include "chai/config.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"

#include <iostream>

#include "umpire/ResourceManager.hpp"

#if defined(CHAI_ENABLE_UM)
#if !defined(CHAI_THIN_GPU_ALLOCATE)
#include <cuda_runtime_api.h>
#endif
#endif

namespace chai {

CHAI_INLINE
void ArrayManager::copy(void * dst, void * src, size_t size) {
   m_resource_manager.copy(dst,src,size);
}

CHAI_INLINE
umpire::Allocator ArrayManager::getAllocator(ExecutionSpace space) {
   return *m_allocators[space];
}

CHAI_INLINE
void ArrayManager::setAllocator(ExecutionSpace space, umpire::Allocator &allocator) {
   *m_allocators[space] = allocator;
}

CHAI_INLINE
bool ArrayManager::syncIfNeeded() {
  if (!m_synced_since_last_kernel) {
     synchronize();
     m_synced_since_last_kernel = true;
     return true;
  }
  return false;
}
} // end of namespace chai

#endif // CHAI_ArrayManager_INL
