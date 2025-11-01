//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ManagedArray_HPP
#define CHAI_ManagedArray_HPP

#include "chai/config.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/Types.hpp"

#include "umpire/Allocator.hpp"

#include <cstddef>

namespace Kripke {

/*!
 * \class ManagedArray
 *
 * \brief Provides an array-like class that automatically transfers data
 * between memory spaces.
 *
 * The ManagedArray class interacts with the ArrayManager to provide a
 * smart-array object that will automatically move its data between different
 * memory spaces on the system. Data motion happens when the copy constructor
 * is called, so the ManagedArray works best when used in co-operation with a
 * programming model like RAJA.
 *
 * \include ./examples/ex1.cpp
 *
 * \tparam T The type of elements stored in the ManagedArray.
 */
template <typename T>
class ManagedArray
{
public:
  using T_non_const = typename std::remove_const<T>::type;

  ManagedArray();

  /*!
   * \brief Copy constructor handles data movement.
   *
   * The copy constructor interacts with the ArrayManager to move the
   * ManagedArray's data between execution spaces.
   *
   * \param other ManagedArray being copied.
   */
  ManagedArray(ManagedArray const& other);

  /*!
   * \brief Allocate data for the ManagedArray in the specified space.
   *
   * The default space for allocations is the CPU.
   *
   * \param elems Number of elements to allocate.
   * \param space Execution space in which to allocate data.
   * \param cback User defined callback for memory events (alloc, free, move)
   */
  void allocate(size_t elems,
                          ExecutionSpace space = CPU,
                          UserCallback const& cback =
                          [] (const PointerRecord*, Action, ExecutionSpace) {});

  /*!
   * \brief Free all data allocated by this ManagedArray.
   */
  //void free(ExecutionSpace space = NONE);

  void move(ExecutionSpace space=NONE,
            bool registerTouch=!std::is_const<T>::value) const;

  /*!
   * \brief Return reference to i-th element of the ManagedArray.
   *
   * \param i Element to return reference to.
   *
   * \return Reference to i-th element.
   */
  template <typename Idx>
  CHAI_HOST_DEVICE T& operator[](const Idx i) const;

  /*!
   * \brief Return the raw pointer to the data in the given execution
   *        space. Optionally move the data to that execution space.
   *
   * \param space The execution space from which to retrieve the raw pointer.
   * \param do_move Ensure data at that pointer is live and valid.
   *
   * @return A copy of the pointer in the given execution space
   */
  T* data(ExecutionSpace space, bool do_move = true) const;

  ManagedArray<T>& operator=(ManagedArray const & other) = default;

  ManagedArray<T>& operator=(ManagedArray && other);

  bool operator==(const ManagedArray<T>& rhs) const;
  bool operator!=(const ManagedArray<T>& from) const;

#ifndef CHAI_DISABLE_RM
  /*!
   * \brief Assign a user-defined callback triggerd upon memory migration.
   *
   * The callback is a function of the form
   *
   *   void callback(chai::ExecutionSpace moved_to, size_t num_bytes)
   *
   * Where moved_to is the execution space that the data was moved to, and
   * num_bytes is the number of bytes moved.
   *
   */
  void setUserCallback(UserCallback const& cback)
  {
    if (m_pointer_record && m_pointer_record != &ArrayManager::s_null_record) {
      m_pointer_record->m_user_callback = cback;
    }
  }
#endif

protected:
  /*!
   * Currently active data pointer.
   */
  mutable T* m_active_pointer = nullptr;
  mutable T* m_active_base_pointer = nullptr;

  /*!
   * Pointer to ArrayManager instance.
   */
  mutable ArrayManager* m_resource_manager = nullptr;

  /*!
   * Number of elements in the ManagedArray.
   */
  mutable size_t m_size = 0;
  mutable size_t m_offset = 0;

  /*!
   * Pointer to PointerRecord data.
   */
  mutable PointerRecord* m_pointer_record = nullptr;

  mutable bool m_is_slice = false;
};

}  // end of namespace chai

#if defined(CHAI_DISABLE_RM)
#include "chai/ManagedArray_thin.inl"
#else
#include "chai/ManagedArray.inl"
#endif
#endif  // CHAI_ManagedArray_HPP
