//
// Copyright (c) 2014-25, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the Kripke/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
  
#ifndef Kripke_ArrayManager_HPP
#define Kripke_ArrayManager_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/PointerRecord.hpp"
#include "chai/Types.hpp"

#if defined(CHAI_ENABLE_RAJA_PLUGIN)
#include "chai/pluginLinker.hpp"
#endif

#include <unordered_map>

#include "umpire/Allocator.hpp"
#include "umpire/util/MemoryMap.hpp"

#if defined(KRIPKE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif
#if defined(KRIPKE_ENABLE_HIP)
#include "hip/hip_runtime_api.h"
#endif

namespace Kripke {

enum class ExecutionSpace : int {
  HOST = 0,
#if defined(KRIPKE_ENABLE_CUDA) || defined(KRIPKE_ENABLE_HIP)
  DEVICE,
#endif
  NUM_EXECUTION_SPACES,
};

enum class MemorySpace : int {
  /*! CPU space */
  CPU = 0,
#if defined(KRIPKE_ENABLE_CUDA) || defined(KRIPKE_ENABLE_HIP)
  /*! GPU space */
  GPU,
#if defined(KRIPKE_ENABLE_UM)
  /*! UVM (Unified Virtual Memory) address space */
  UVM,
#endif
#if defined(KRIPKE_ENABLE_MI300_UNIFIEDMEMORY)
  /*! MI300 CPU/GPU single address space */
  GPUONLY,
#endif
#endif
  /*! Used to count total number of spaces */
  NUM_MEMORY_SPACES,
};

class UmpireMemorySpace
{
protected:
   umpire::ResourceManager &rm;
   umpire::Allocator allocator;
   bool owns_allocator{false};

public:
   UmpireMemorySpace(const char * name, const char * space)
      : rm(umpire::ResourceManager::getInstance())
   {
      if (!rm.isAllocator(name))
      {
         allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
                        name, rm.getAllocator(space));
         owns_allocator = true;
      }
      else
      {
         allocator = rm.getAllocator(name);
         owns_allocator = false;
      }
   }
   ~UmpireMemorySpace() { if (owns_allocator) { allocator.release(); } }
};

/*!
 * \brief Singleton that manages caching and movement of ManagedArray objects.
 *
 * The ArrayManager class co-ordinates the allocation and movement of
 * ManagedArray objects. These objects are cached, and data is only copied
 * between ExecutionSpaces when necessary. This functionality is typically
 * hidden behind a programming model layer, such as RAJA, or the exmaple
 * included in util/forall.hpp
 *
 * The ArrayManager is a singleton, so must always be accessed through the
 * static getInstance method. Here is an example using the ArrayManager:
 *
 * \code
 * const chai::ArrayManager* rm = chai::ArrayManager::getInstance();
 * rm->setExecutionSpace(chai::CPU);
 * // Do something in with ManagedArrays on the CPU... but they must be copied!
 * rm->setExecutionSpace(chai::NONE);
 * \endcode
 */
class ArrayManager
{
public:
  template <typename T>
  using T_non_const = typename std::remove_const<T>::type;

  using PointerMap = umpire::util::MemoryMap<PointerRecord*>;

  static PointerRecord s_null_record;

  /*!
   * \brief Get the singleton instance.
   *
   * \return Pointer to the ArrayManager instance.
   *
   */
  static ArrayManager* getInstance();

  /*!
   * \brief Set the current execution space.
   *
   * \param space The space to set as current.
   */
  void setExecutionSpace(ExecutionSpace space);

  /*!
   * \brief Get the current execution space.
   *
   * \return The current execution space.jo
   */
  ExecutionSpace getExecutionSpace();

  /*!
   * \brief Move data in pointer to the current execution space.
   *
   * \param pointer Pointer to data in any execution space.
   * \return Pointer to data in the current execution space.
   */
  void* move(void* pointer,
             PointerRecord* pointer_record,
             ExecutionSpace = NONE);

  /*!
   * \brief Register a touch of the pointer in the current execution space.
   *
   * \param pointer Raw pointer to register a touch of.
   */
  void registerTouch(PointerRecord* pointer_record);

  /*!
   * \brief Register a touch of the pointer in the given execution space.
   *
   * The pointer doesn't need to exist in the space being touched.
   *
   * \param pointer Raw pointer to register a touch of.
   * \param space Space to register touch.
   */
  void registerTouch(PointerRecord* pointer_record, ExecutionSpace space);

  /*!
   * \brief Make a new allocation of the data described by the PointerRecord in
   * the given space.
   *
   * \param pointer_record
   * \param space Space in which to make the allocation.
   */
  void allocate(PointerRecord* pointer_record, ExecutionSpace space = CPU);

  /*!
   * \brief Set the default space for new ManagedArray allocations.
   *
   * ManagedArrays allocated without an explicit ExecutionSpace argument will
   * be allocated in space after this routine is called.
   *
   * \param space New space for default allocations.
   */
  void setDefaultAllocationSpace(ExecutionSpace space);

  /*!
   * \brief Get the currently set default allocation space.
   *
   * See also setDefaultAllocationSpace.
   *
   * \return Current default space for allocations.
   */
  ExecutionSpace getDefaultAllocationSpace();

  /*!
   * \brief Free allocation(s) associated with the given PointerRecord.
   *        Default (space == NONE) will free all allocations and delete
   *        the pointer record.
   */
  void free(PointerRecord* pointer, ExecutionSpace space = NONE);

  /*!
   * \brief Get the size of the given pointer.
   *
   * \param pointer Pointer to find the size of.
   * \return Size of pointer.
   */
  size_t getSize(void* pointer);

  PointerRecord* makeManaged(void* pointer,
                             size_t size,
                             ExecutionSpace space,
                             bool owned);

  /*!
   * \brief Set touched to false in all spaces for the given PointerRecord.
   *
   * \param pointer_record PointerRecord to reset.
   */
  void resetTouch(PointerRecord* pointer_record);

  /*!
   * \brief Find the PointerRecord corresponding to the raw pointer.
   *
   * \param pointer Raw pointer to find the PointerRecord for.
   *
   * \return PointerRecord containing the raw pointer, or an empty
   *         PointerRecord if none found.
   */
  PointerRecord* getPointerRecord(void* pointer);

  /*!
   * \brief Create a copy of the pointer map.
   *
   * \return A copy of the pointer map. Can be used to find memory leaks.
   */
  std::unordered_map<void*, const PointerRecord*> getPointerMap() const;

  /*!
   * \brief Get the allocator ID
   *
   * \return The allocator ID.
   */
  int getAllocatorId(ExecutionSpace space) const;

  /*!
   * \brief Wraps our resource manager's copy.
   */
  void copy(void * dst, void * src, size_t size); 
  
  /*!
   * \brief Registering an allocation with the ArrayManager
   *
   * \param record PointerRecord of this allocation.
   * \param space Space in which the pointer was allocated.
   * \param owned Should the allocation be free'd by CHAI?
   */
  void registerPointer(PointerRecord* record,
                                         ExecutionSpace space,
                                         bool owned = true);

  /*!
   * \brief Deregister a PointerRecord from the ArrayManager.
   *
   * \param record PointerRecord of allocation to deregister.
   * \param deregisterFromUmpire If true, deregister from umpire as well.
   */
  void deregisterPointer(PointerRecord* record, bool deregisterFromUmpire=false);

  /*!
   * \brief set the allocator for an execution space.
   *
   * \param space Execution space to set the default allocator for.
   * \param allocator The allocator to use for this space. Will be copied into chai.
   */
  void setAllocator(ExecutionSpace space, umpire::Allocator &allocator);

  /*!
   * \brief Get the allocator for an execution space.
   *
   * \param space Execution space of the allocator to get.
   *
   * \return The allocator for the given space.
   */
  umpire::Allocator getAllocator(ExecutionSpace space);
  
  /*!
   * \brief synchronize the device if there hasn't been a synchronize since the last kernel
   */
  bool syncIfNeeded();

  static const char * h_umpire_name;
  static const char * d_umpire_name;
  static const char * um_umpire_name;
  static const char * GetUmpireHostAllocatorName() { return h_umpire_name; }
  static void SetUmpireHostAllocatorName(const char * h_name) { h_umpire_name = h_name; }
  static const char * GetUmpireDeviceAllocatorName() { return d_umpire_name; }
  static void SetUmpireDeviceAllocatorName(const char * d_name) { d_umpire_name = d_name; }
  static const char * GetUmpireUMAllocatorName() { return um_umpire_name; }
  static void SetUmpireUMAllocatorName(const char * h_name) { um_umpire_name = h_name; }

protected:
  /*!
   * \brief Construct a new ArrayManager.
   *
   * The constructor is a protected member, ensuring that it can
   * only be called by the singleton getInstance method.
   */
  ArrayManager();



private:


  /*!
   * \brief Move data in PointerRecord to the corresponding ExecutionSpace.
   *
   * \param record
   * \param space
   */
  void move(PointerRecord* record, ExecutionSpace space);
  
  /*!
   * Current execution space.
   */
  static thread_local ExecutionSpace m_current_execution_space;

  /**
   * Default space for new allocations.
   */
  ExecutionSpace m_default_allocation_space;

  /*!
   * Map of active ManagedArray pointers to their corresponding PointerRecord.
   */
  PointerMap m_pointer_map;

  /*!
   *
   * \brief Array of umpire::Allocators, indexed by ExecutionSpace.
   */
  UmpireMemorySpace* m_allocators[NUM_EXECUTION_SPACES];

  /*!
   * \brief The umpire resource manager.
   */
  umpire::ResourceManager& m_resource_manager;

  /*!
   * \brief Used for thread-safe operations.
   */
  mutable std::mutex m_mutex;

  /*!
   * Whether or not a synchronize has been performed since the launch of the last
   * GPU context
   */
  static thread_local bool m_synced_since_last_kernel;

};

}  // end of namespace chai

#include "chai/ArrayManager.inl"

#endif  // CHAI_ArrayManager_HPP
