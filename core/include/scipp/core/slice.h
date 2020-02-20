// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Owen Arnold
#include "scipp-core_export.h"
#include "scipp/common/index.h"
#include "scipp/units/unit.h"

#ifndef SCIPP_CORE_SLICE_H
#define SCIPP_CORE_SLICE_H

namespace scipp {
namespace core {

/// Describes a slice to make over a dimension either as a single index or as a
/// range
class SCIPP_CORE_EXPORT Slice {

public:
  Slice(const Dim dim_, const scipp::index begin_, const scipp::index end_);
  Slice(const Dim dim_, const scipp::index begin_);
  Slice &operator=(const Slice &) = default;
  bool operator==(const Slice &other) const noexcept;
  bool operator!=(const Slice &other) const noexcept;
  scipp::index begin() const noexcept { return m_begin; };
  scipp::index end() const noexcept { return m_end; };
  Dim dim() const noexcept { return m_dim; };
  bool isRange() const noexcept { return m_end != -1; };

private:
  Dim m_dim;
  scipp::index m_begin;
  scipp::index m_end;
};

} // namespace core
} // namespace scipp

#endif // SCIPP_CORE_SLICE_H
