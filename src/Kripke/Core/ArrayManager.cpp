//
// Copyright (c) 2014-25, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the Kripke/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
  
#include "chai/ArrayManager.hpp"

#include "chai/config.hpp"

#include "umpire/ResourceManager.hpp"

namespace chai
{
thread_local ExecutionSpace ArrayManager::m_current_execution_space;
thread_local bool ArrayManager::m_synced_since_last_kernel = false;

PointerRecord ArrayManager::s_null_record = PointerRecord();

ArrayManager* ArrayManager::getInstance()
{
  static ArrayManager s_resource_manager_instance;
  return &s_resource_manager_instance;
}

ArrayManager::ArrayManager() :
  m_pointer_map{},
  m_allocators{},
  m_resource_manager{umpire::ResourceManager::getInstance()}
{
  m_pointer_map.clear();
  m_current_execution_space = HOST;
  m_default_allocation_space = HOST;

  m_allocators[CPU] = new UmpireMemorySpace(ArrayManager::GetUmpireHostAllocatorName(), "HOST");

#if defined(KRIPKE_ENABLE_CUDA) || defined(KRIPKE_ENABLE_HIP)
  m_allocators[GPU] = new UmpireMemorySpace(ArrayManager::GetUmpireDeviceAllocatorName(), "DEVICE");
#endif

#if defined(KRIPKE_ENABLE_UM)
  m_allocators[UM] = new UmpireMemorySpace(ArrayManager::GetUmpireUMAllocatorName(), "UM");
#endif

#if defined(KRIPKE_ENABLE_MI300_UNIFIEDMEMORY)
  m_allocators[GPUONLY] = new UmpireMemorySpace(ArrayManager::GetUmpireDeviceAllocatorName(), "DEVICE");
#endif
}

void ArrayManager::registerPointer(
   PointerRecord* record,
   ExecutionSpace space,
   bool owned)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  auto pointer = record->m_pointers[space];

  auto found_pointer_record_pair = m_pointer_map.find(pointer);
  if (found_pointer_record_pair != m_pointer_map.end()) {
     PointerRecord ** found_pointer_record_addr = found_pointer_record_pair->second;
     if (found_pointer_record_addr != nullptr) {

        PointerRecord *foundRecord = *found_pointer_record_addr;
        if (foundRecord != record) {
           for (int fspace = CPU; fspace < NUM_EXECUTION_SPACES; ++fspace) {
              foundRecord->m_pointers[fspace] = nullptr;
           }

           delete foundRecord;
        }
     }
  }

  m_pointer_map.insert(pointer, record);

  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    if (!record->m_pointers[i]) record->m_owned[i] = true;
  }
  record->m_owned[space] = owned;

  if (pointer) {
     // if umpire already knows about this pointer, we want to make sure its records and ours
     // are consistent
     if (m_resource_manager.hasAllocator(pointer)) {
         umpire::util::AllocationRecord *allocation_record = const_cast<umpire::util::AllocationRecord *>(m_resource_manager.findAllocationRecord(pointer));
         allocation_record->size = record->m_size;
     }
     // register with umpire if it's not there so that umpire can perform data migrations
     else {
        umpire::util::AllocationRecord new_allocation_record;
        new_allocation_record.ptr = pointer;
        new_allocation_record.size = record->m_size;
        new_allocation_record.strategy = m_resource_manager.getAllocator(record->m_allocators[space]).getAllocationStrategy();

        m_resource_manager.registerAllocation(pointer, new_allocation_record);
     }
  }
}

void ArrayManager::setExecutionSpace(ExecutionSpace space)
{
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
   if (isGPUSimMode() && chai::NONE != space) {
      space = chai::GPU;
   }
#endif

  if (chai::GPU == space) {
    m_synced_since_last_kernel = false;
  }

#if defined(CHAI_THIN_GPU_ALLOCATE)
 if (chai::CPU == space) {
    syncIfNeeded();
 }
#endif

  m_current_execution_space = space;
}

void* ArrayManager::move(void* pointer,
                         PointerRecord* pointer_record,
                         ExecutionSpace space)
{
  // Check for default arg (NONE)
  if (space == NONE) {
    space = m_current_execution_space;
  }

  if (space == NONE) {
    return pointer;
  }

  move(pointer_record, space);

  return pointer_record->m_pointers[space];
}

ExecutionSpace ArrayManager::getExecutionSpace()
{
  return m_current_execution_space;
}

void ArrayManager::registerTouch(PointerRecord* pointer_record)
{
  registerTouch(pointer_record, m_current_execution_space);
}

void ArrayManager::registerTouch(PointerRecord* pointer_record,
                                 ExecutionSpace space)
{
  if (pointer_record && pointer_record != &s_null_record) {
     if (space != NONE) {
       pointer_record->m_touched[space] = true;
       pointer_record->m_last_space = space;
     }
  }
}


void ArrayManager::resetTouch(PointerRecord* pointer_record)
{
  if (pointer_record && pointer_record!= &s_null_record) {
    for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
      pointer_record->m_touched[space] = false;
    }
  }
}


/* Not all GPU platform runtimes (notably HIP), will give you asynchronous copies to the device by default, so we leverage
 * umpire's API for asynchronous copies using camp resources in this method, based off of the CHAI destination space
 * */
static void copy(void * dst_pointer, void * src_pointer, umpire::ResourceManager & manager, ExecutionSpace dst_space, ExecutionSpace src_space) {

#ifdef CHAI_ENABLE_CUDA
   camp::resources::Resource device_resource(camp::resources::Cuda::get_default());
#elif defined(CHAI_ENABLE_HIP)
   camp::resources::Resource device_resource(camp::resources::Hip::get_default());
#else
   camp::resources::Resource device_resource(camp::resources::Host::get_default());
#endif

   camp::resources::Resource host_resource(camp::resources::Host::get_default());
   if (dst_space == GPU || src_space == GPU) {
      // Do the copy using the device resource
      manager.copy(dst_pointer, src_pointer, device_resource);
   } else {
      // Do the copy using the host resource
      manager.copy(dst_pointer, src_pointer, host_resource);
   }
   // Ensure device to host copies are synchronous
   if (dst_space == CPU && src_space == GPU) {
      device_resource.wait();
   }
}

void ArrayManager::move(PointerRecord* record, ExecutionSpace space)
{
  if (space == NONE) {
    return;
  }

  if (space == record->m_last_space) {
    return;
  }

#if defined(CHAI_ENABLE_UM)
  if (record->m_last_space == UM) {
    return;
  }
#endif

#if defined(KRIPKE_ENABLE_MI300_UNIFIEDMEMORY)
  if (record->m_last_space == GPUONLY) {
    return;
  }
#endif

  ExecutionSpace prev_space = record->m_last_space;

  void* src_pointer = record->m_pointers[prev_space];
  void* dst_pointer = record->m_pointers[space];

  if (!dst_pointer) {
    allocate(record, space);
    dst_pointer = record->m_pointers[space];
  }


  if ( (!record->m_touched[record->m_last_space]) || (! src_pointer )) {
    return;
  } else if (dst_pointer != src_pointer) {
    // Exclude the copy if src and dst are the same (can happen for PINNED memory)
    {
      chai::copy(dst_pointer, src_pointer, m_resource_manager, space, prev_space);
    }

  }

  resetTouch(record);
}

void ArrayManager::allocate(
    PointerRecord* pointer_record,
    ExecutionSpace space)
{
  auto size = pointer_record->m_size;
  auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[space]);

  pointer_record->m_pointers[space] = alloc.allocate(size);
  registerPointer(pointer_record, space);
}

void ArrayManager::free(PointerRecord* pointer_record, ExecutionSpace spaceToFree)
{
  if (!pointer_record) return;

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (space == spaceToFree || spaceToFree == NONE) {
      if (pointer_record->m_pointers[space]) {
        void* space_ptr = pointer_record->m_pointers[space];
        if (pointer_record->m_owned[space]) {
#if defined(CHAI_ENABLE_UM)
          if (space_ptr == pointer_record->m_pointers[UM]) {

            auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[UM]);
            alloc.deallocate(space_ptr);

            for (int space_t = CPU; space_t < NUM_EXECUTION_SPACES; ++space_t) {
              if (space_ptr == pointer_record->m_pointers[space_t]) {
                pointer_record->m_pointers[space_t] = nullptr;
              }
            }
          } else
#endif
#if defined(CHAI_ENABLE_PINNED)
          if (space_ptr == pointer_record->m_pointers[PINNED]) {

            auto alloc = m_resource_manager.getAllocator(
                pointer_record->m_allocators[PINNED]);
            alloc.deallocate(space_ptr);

            for (int space_t = CPU; space_t < NUM_EXECUTION_SPACES; ++space_t) {
              if (space_ptr == pointer_record->m_pointers[space_t]) {
                pointer_record->m_pointers[space_t] = nullptr;
              }
            }
          } else
#endif
          {

            auto alloc = m_resource_manager.getAllocator(
                pointer_record->m_allocators[space]);
            alloc.deallocate(space_ptr);

            pointer_record->m_pointers[space] = nullptr;
          }
        }
        else
        {
          m_resource_manager.deregisterAllocation(space_ptr);
        }
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_pointer_map.erase(space_ptr);
        }
      }
    }
  }
  
  if (pointer_record != &s_null_record && spaceToFree == NONE) {
    delete pointer_record;
  }
}

void ArrayManager::setDefaultAllocationSpace(ExecutionSpace space)
{
  m_default_allocation_space = space;
}

ExecutionSpace ArrayManager::getDefaultAllocationSpace()
{
  return m_default_allocation_space;
}

PointerRecord* ArrayManager::getPointerRecord(void* pointer)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  auto record = m_pointer_map.find(pointer);
  return record->second ? *record->second : &s_null_record;
}

std::unordered_map<void*, const PointerRecord*>
ArrayManager::getPointerMap() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::unordered_map<void*, const PointerRecord*> mapCopy;

  for (const auto& entry : m_pointer_map) {
    mapCopy[entry.first] = *entry.second;
  }

  return mapCopy;
}

int
ArrayManager::getAllocatorId(ExecutionSpace space) const
{
  return m_allocators[space]->getId();
}

}  // end of namespace chai
