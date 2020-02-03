// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef SCIPP_COMMON_NUMERIC_H
#define SCIPP_COMMON_NUMERIC_H

#include <algorithm>
#include <cmath>
#include <limits>

#include "scipp/common/index.h"

namespace scipp::numeric {

template <class Range> bool is_linspace(const Range &range) {
  if (scipp::size(range) < 2)
    return false;
  if (range.back() <= range.front())
    return false;

  using T = typename Range::value_type;
  const T delta = (range.back() - range.front()) / (scipp::size(range) - 1);
  constexpr int32_t ulp = 4;
  const T epsilon =
      std::numeric_limits<T>::epsilon() * (range.front() + range.back()) * ulp;

  return std::adjacent_find(range.begin(), range.end(),
                            [epsilon, delta](const auto &a, const auto &b) {
                              return std::abs(std::abs(b - a) - delta) >
                                     epsilon;
                            }) == range.end();
}

/// Compute the ratio of a geometric sequence of numbers
template <class Range> auto geometric_ratio(const Range &range) {
  return std::exp(std::log(range.back() / range.front()) /
                  (scipp::size(range) - 1));
  // return std::pow(range.back() / range.front(), 1.0 /
  //                 (scipp::size(range) - 1));
}

template <class Range> bool is_logspace(const Range &range) {
  if (scipp::size(range) < 2)
    return false;
  if (range.back() <= range.front())
    return false;

  using T = typename Range::value_type;
  if (range.front() == static_cast<T>(0.))
    return false;
  const T delta = geometric_ratio(range);
  constexpr int32_t ulp = 4;
  const T epsilon = std::numeric_limits<T>::epsilon() * delta * ulp;

  return std::adjacent_find(range.begin(), range.end(),
                            [epsilon, delta](const auto &a, const auto &b) {
                              return std::abs(std::abs(b / a) - delta) >
                                     epsilon;
                            }) == range.end();
}

} // namespace scipp::numeric

#endif // SCIPP_COMMON_NUMERIC_H
