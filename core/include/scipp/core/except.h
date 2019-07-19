// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef EXCEPT_H
#define EXCEPT_H

#include <stdexcept>
#include <string>

#include "scipp-core_export.h"
#include "scipp/core/dtype.h"
#include "scipp/core/index.h"
#include "scipp/units/unit.h"

namespace scipp::core {

class DataConstProxy;
class DatasetConstProxy;
class Dataset;
class Dimensions;
class Variable;
class VariableConstProxy;
struct Slice;

SCIPP_CORE_EXPORT std::string to_string(const DType dtype);
SCIPP_CORE_EXPORT std::string to_string(const Dimensions &dims);
SCIPP_CORE_EXPORT std::string to_string(const Slice &slice);
SCIPP_CORE_EXPORT std::string to_string(const units::Unit &unit);
SCIPP_CORE_EXPORT std::string to_string(const Variable &variable);
SCIPP_CORE_EXPORT std::string to_string(const VariableConstProxy &variable);
SCIPP_CORE_EXPORT std::string to_string(const DataConstProxy &data);
SCIPP_CORE_EXPORT std::string to_string(const Dataset &dataset);
SCIPP_CORE_EXPORT std::string to_string(const DatasetConstProxy &dataset);

template <class T> std::string array_to_string(const T &arr);

template <class T> std::string element_to_string(const T &item) {
  using std::to_string;
  if constexpr (std::is_same_v<T, std::string>)
    return {'"' + item + "\", "};
  else if constexpr (std::is_same_v<T, Eigen::Vector3d>)
    return {"(" + to_string(item[0]) + ", " + to_string(item[1]) + ", " +
            to_string(item[2]) + "), "};
  else if constexpr (is_sparse_v<T>)
    return array_to_string(item) + ", ";
  else if constexpr (std::is_same_v<T, Dataset>)
    return {"Dataset, "};
  else
    return to_string(item) + ", ";
}

template <class T> std::string array_to_string(const T &arr) {
  const auto size = scipp::size(arr);
  if (size == 0)
    return std::string("[]");
  std::string s = "[";
  for (scipp::index i = 0; i < scipp::size(arr); ++i) {
    if (i == 4 && size > 8) {
      s += "..., ";
      i = size - 4;
    }
    s += element_to_string(arr[i]);
  }
  s.resize(s.size() - 2);
  s += "]";
  return s;
}

namespace except {

struct SCIPP_CORE_EXPORT TypeError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SCIPP_CORE_EXPORT DimensionError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SCIPP_CORE_EXPORT DimensionMismatchError : public DimensionError {
  DimensionMismatchError(const Dimensions &expected, const Dimensions &actual);
};

struct SCIPP_CORE_EXPORT DimensionNotFoundError : public DimensionError {
  DimensionNotFoundError(const Dimensions &expected, const Dim actual);
};

struct SCIPP_CORE_EXPORT DimensionLengthError : public DimensionError {
  DimensionLengthError(const Dimensions &expected, const Dim actual,
                       const scipp::index length);
};

struct SCIPP_CORE_EXPORT SparseDimensionError : public DimensionError {
  SparseDimensionError()
      : DimensionError("Unsupported operation for sparse dimensions.") {}
};

struct SCIPP_CORE_EXPORT DatasetError : public std::runtime_error {
  DatasetError(const Dataset &dataset, const std::string &message);
  DatasetError(const DatasetConstProxy &dataset, const std::string &message);
};

struct SCIPP_CORE_EXPORT VariableError : public std::runtime_error {
  VariableError(const Variable &variable, const std::string &message);
  VariableError(const VariableConstProxy &variable, const std::string &message);
};

struct SCIPP_CORE_EXPORT VariableMismatchError : public VariableError {
  template <class A, class B>
  VariableMismatchError(const A &a, const B &b)
      : VariableError(a, "expected to match\n" + to_string(b)) {}
};

struct SCIPP_CORE_EXPORT UnitError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SCIPP_CORE_EXPORT SizeError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SCIPP_CORE_EXPORT UnitMismatchError : public UnitError {
  UnitMismatchError(const units::Unit &a, const units::Unit &b);
};

struct SCIPP_CORE_EXPORT SliceError : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

struct SCIPP_CORE_EXPORT CoordMismatchError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SCIPP_CORE_EXPORT VariancesError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

} // namespace except

namespace expect {
template <class A, class B> void variablesMatch(const A &a, const B &b) {
  if (a != b)
    throw except::VariableMismatchError(a, b);
}
SCIPP_CORE_EXPORT void dimensionMatches(const Dimensions &dims, const Dim dim,
                                        const scipp::index length);

template <class T, class... Ts>
void sizeMatches(const T &range, const Ts &... other) {
  if (((scipp::size(range) != scipp::size(other)) || ...))
    throw except::SizeError("Expected matching sizes.");
}
SCIPP_CORE_EXPORT void equals(const units::Unit &a, const units::Unit &b);
SCIPP_CORE_EXPORT void equals(const Dimensions &a, const Dimensions &b);

template <class T> void contains(const T &a, const T &b) {
  if (!a.contains(b))
    throw std::runtime_error("Expected " + to_string(a) + " to contain " +
                             to_string(b) + ".");
}
template <class T> void unit(const T &object, const units::Unit &unit) {
  expect::equals(object.unit(), unit);
}

template <class T> void countsOrCountsDensity(const T &object) {
  if (!object.unit().isCounts() && !object.unit().isCountDensity())
    throw except::UnitError("Expected counts or counts-density, got " +
                            object.unit().name() + '.');
}

void SCIPP_CORE_EXPORT validSlice(const Dimensions &dims, const Slice &slice);

template <typename T> void coordsAndLabelsMatch(const T &a, const T &b) {
  if (a.coords() != b.coords() || a.labels() != b.labels())
    throw except::CoordMismatchError("Expected coords and labels to match.");

  for (const auto & [ name, item ] : b) {
    if (a.contains(name)) {
      if (item.dims().sparse()) {
        const auto sparseDim = item.dims().sparseDim();

        const auto coords_a = a[name].coords();
        const auto coords_b = item.coords();

        /* Fail if: */
        /* - presence of a sparse dimension is not identical in both items */
        /* - both items have a sparse dimension but it's coordinates differ */
        if ((coords_a.contains(sparseDim) != coords_b.contains(sparseDim)) ||
            (coords_a.contains(sparseDim) &&
             coords_a[sparseDim] != coords_b[sparseDim])) {
          throw except::CoordMismatchError("Expected sparse coords to match.");
        }

        /* Check that a and b have identical sparse labels */
        const auto labels_a = a[name].labels();
        for (const auto & [ label_name, label_b ] : item.labels()) {
          if (!labels_a.contains(label_name)) {
            throw except::CoordMismatchError(
                "Expected sparse labels to match.");
          }

          if (labels_a[label_name] != label_b) {
            throw except::CoordMismatchError(
                "Expected sparse labels to match.");
          }
        }
      }
    }
  }
}

void SCIPP_CORE_EXPORT coordsAndLabelsAreSuperset(const DataConstProxy &a,
                                                  const DataConstProxy &b);
void SCIPP_CORE_EXPORT notSparse(const Dimensions &dims);
void SCIPP_CORE_EXPORT validDim(const Dim dim);
void SCIPP_CORE_EXPORT validExtent(const scipp::index size);

} // namespace expect
} // namespace scipp::core

#endif // EXCEPT_H
