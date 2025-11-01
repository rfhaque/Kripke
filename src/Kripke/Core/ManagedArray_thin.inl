//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ManagedArray_thin_INL
#define CHAI_ManagedArray_thin_INL

#include "ManagedArray.hpp"

#if defined(CHAI_ENABLE_UM)
#if !defined(CHAI_THIN_GPU_ALLOCATE)
#include <cuda_runtime_api.h>
#endif
#endif

namespace chai {

template<typename T>
CHAI_INLINE
ManagedArray<T>::ManagedArray() = default;

template<typename T>
CHAI_INLINE
ManagedArray<T>::ManagedArray(ManagedArray const& other) = default;

template <typename T>
T* ManagedArray<T>::data(ExecutionSpace space, bool do_move) const
{
#if defined(CHAI_THIN_GPU_ALLOCATE)
  if (do_move && space != chai::GPU) {
      ArrayManager::getInstance()->syncIfNeeded();
  }
#endif
  return m_active_pointer;
}

template<typename T>
CHAI_INLINE
void ManagedArray<T>::allocate(size_t elems,
                               ExecutionSpace space,
                               UserCallback const &) {
  if (!m_is_slice) {
     if (elems > 0) {
       (void) space; // Quiet compiler warning when CHAI_LOG does nothing
       CHAI_LOG(Debug, "Allocating array of size " << elems
                                                   << " in space "
                                                   << space);

       m_size = elems*sizeof(T);

       m_active_pointer = (T*) chai::ArrayManager::getInstance()->getAllocator(chai::GPU).allocate(m_size);

       CHAI_LOG(Debug, "m_active_ptr allocated at address: " << m_active_pointer);
     
     }
     else {
        m_active_pointer = nullptr;
        m_size = 0;
     }
  }
  m_active_base_pointer = m_active_pointer;
}

/*
template <typename T>
CHAI_INLINE void ManagedArray<T>::free(ExecutionSpace space)
{
  if (!m_is_slice) {
    if (space == CPU || space == NONE) {
#if defined(CHAI_THIN_GPU_ALLOCATE)
      if (m_active_pointer) {
         auto allocator = chai::ArrayManager::getInstance()->getAllocator(chai::GPU);
         allocator.deallocate((void *)m_active_pointer);
      }
#elif defined(CHAI_ENABLE_UM)
      chai::gpuFree(m_active_pointer);
#else
      ::free((void *)m_active_pointer);
#endif
      m_active_pointer = nullptr;
      m_active_base_pointer = nullptr;
      m_size = 0;
    }
  }
  else {
    CHAI_LOG(Debug, "tried to free slice!");
  }
}
*/

template <typename T>
CHAI_INLINE void ManagedArray<T>::move(ExecutionSpace, bool) const
{
}

template <typename T>
template <typename Idx>
CHAI_INLINE CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const Idx i) const
{
  return m_active_pointer[i];
}

template<typename T>
CHAI_INLINE
ManagedArray<T>&
ManagedArray<T>::operator= (ManagedArray && other) {
  if (this != &other) {
      *this = other;
      other = nullptr;
  }
  return *this;
}

template <typename T>
CHAI_INLINE bool ManagedArray<T>::operator==(
    const ManagedArray<T>& rhs) const
{
  return (m_active_pointer == rhs.m_active_pointer);
}

template <typename T>
CHAI_INLINE bool ManagedArray<T>::operator!=(
    const ManagedArray<T>& rhs) const
{
  return (m_active_pointer != rhs.m_active_pointer);
}

}  // end of namespace chai

#endif  // CHAI_ManagedArray_thin_INL
