//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ManagedArray_INL
#define CHAI_ManagedArray_INL

#include "ManagedArray.hpp"
#include "ArrayManager.hpp"

namespace chai {

template<typename T>
CHAI_INLINE
ManagedArray<T>::ManagedArray():
  m_active_pointer(nullptr),
  m_active_base_pointer(nullptr),
  m_resource_manager(nullptr),
  m_size(0),
  m_offset(0),
  m_pointer_record(nullptr),
  m_is_slice(false)
{
#if !defined(CHAI_DEVICE_COMPILE)
  m_resource_manager = ArrayManager::getInstance();
  m_pointer_record = &ArrayManager::s_null_record;
#endif
}

template<typename T>
CHAI_INLINE
ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_active_pointer(other.m_active_pointer),
  m_active_base_pointer(other.m_active_base_pointer),
  m_resource_manager(other.m_resource_manager),
  m_size(other.m_size),
  m_offset(other.m_offset),
  m_pointer_record(other.m_pointer_record),
  m_is_slice(other.m_is_slice)
{
  if (m_active_base_pointer || m_size > 0 ) {
     // we only update m_size if we are not null and we have a pointer record
     if (m_pointer_record && !m_is_slice) {
        m_size = m_pointer_record->m_size;
     }
     move(m_resource_manager->getExecutionSpace());
  }
}

template<typename T>
void ManagedArray<T>::allocate(
    size_t elems,
    ExecutionSpace space, 
    const UserCallback& cback) 
{
  if(!m_is_slice) {
     if (elems > 0) {
       if (m_pointer_record == &ArrayManager::s_null_record) {
         // since we are about to allocate, this will get registered
         m_pointer_record = new PointerRecord();
         for (int s = CPU; s < NUM_EXECUTION_SPACES; ++s) {
           ExecutionSpace allocator_space = ExecutionSpace(s);
           m_pointer_record->m_allocators[s] = m_resource_manager->getAllocatorId(allocator_space);
         }
       }

       m_pointer_record->m_user_callback = cback;
       m_size = elems*sizeof(T);
       m_pointer_record->m_size = m_size;

       if (space != NONE) {
         m_resource_manager->allocate(m_pointer_record, space);
         m_active_base_pointer = static_cast<T*>(m_pointer_record->m_pointers[space]);
       } else {
         m_active_base_pointer = nullptr;
         m_pointer_record->m_pointers[space] = nullptr;
       }
       m_active_pointer = m_active_base_pointer; // Cannot be a slice

    }
  }
}

/*
template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free(ExecutionSpace space)
{
  if(!m_is_slice) {
    if (m_resource_manager == nullptr) {
       m_resource_manager = ArrayManager::getInstance();
    }
    if (m_pointer_record == &ArrayManager::s_null_record) {
       m_pointer_record = m_resource_manager->makeManaged((void *)m_active_base_pointer,m_size,space,true);
    }
    m_resource_manager->free(m_pointer_record, space);
    m_active_pointer = nullptr;
    m_active_base_pointer = nullptr;

    m_size = 0;
    m_offset = 0;
    // The call to m_resource_manager::free, above, has deallocated m_pointer_record if space == NONE.
    if (space == NONE) {
       m_pointer_record = &ArrayManager::s_null_record;
    }
  } else {
    CHAI_LOG(Debug, "Cannot free a slice!");
  }
}
*/

template <typename T>
CHAI_INLINE
void ManagedArray<T>::move(ExecutionSpace space, bool registerTouch) const
{
  if (m_pointer_record != &ArrayManager::s_null_record) {
     CHAI_LOG(Debug, "Moving " << m_active_pointer);
     m_active_base_pointer = static_cast<T*>(m_resource_manager->move((void *)m_active_base_pointer, m_pointer_record, space));
     m_active_pointer = m_active_base_pointer + m_offset;

     CHAI_LOG(Debug, "Moved to " << m_active_pointer);
     if (registerTouch) {
       CHAI_LOG(Debug, "T is non-const, registering touch of pointer" << m_active_pointer);
       m_resource_manager->registerTouch(m_pointer_record, space);
     }
   }
}

template<typename T>
template<typename Idx>
CHAI_INLINE
CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const Idx i) const {
  return m_active_pointer[i];
}

template<typename T>
T* ManagedArray<T>::data(ExecutionSpace space, bool do_move) const {
   if (m_pointer_record == nullptr || m_pointer_record == &ArrayManager::s_null_record) {
      return nullptr;
   }

   if (m_size == 0 && !m_is_slice) {
      return nullptr;
   }

   if (do_move) {
      ExecutionSpace oldContext = m_resource_manager->getExecutionSpace();
      m_resource_manager->setExecutionSpace(space);
      move(space);
      m_resource_manager->setExecutionSpace(oldContext);
   }

   int offset = m_is_slice ? m_offset : 0 ;
   return ((T*) m_pointer_record->m_pointers[space]) + offset;
}

template<typename T>
CHAI_INLINE
ManagedArray<T>&
ManagedArray<T>::operator= (ManagedArray && other) {
  *this = other;
  other = nullptr;
  return *this;
}

template<typename T>
CHAI_INLINE
bool
ManagedArray<T>::operator== (const ManagedArray<T>& rhs) const
{
  return (m_active_pointer ==  rhs.m_active_pointer);
}

template<typename T>
CHAI_INLINE
bool
ManagedArray<T>::operator!= (const ManagedArray<T>& rhs) const
{
  return (m_active_pointer !=  rhs.m_active_pointer);
}

} // end of namespace chai

#endif // CHAI_ManagedArray_INL
