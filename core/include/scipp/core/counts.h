// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef SCIPP_CORE_COUNTS_H
#define SCIPP_CORE_COUNTS_H

#include <vector>

#include "scipp-core_export.h"
#include "scipp/core/dataset.h"
#include "scipp/units/unit.h"

namespace scipp::core {

namespace counts {
SCIPP_CORE_EXPORT std::vector<Variable>
getBinWidths(const CoordsConstView &c, const std::vector<Dim> &dims);

SCIPP_CORE_EXPORT void toDensity(const DataArrayView data,
                                 const std::vector<Variable> &binWidths);
SCIPP_CORE_EXPORT Dataset toDensity(Dataset d, const Dim dim);
SCIPP_CORE_EXPORT Dataset toDensity(Dataset d, const std::vector<Dim> &dims);
SCIPP_CORE_EXPORT DataArray toDensity(DataArray a, const Dim dim);
SCIPP_CORE_EXPORT DataArray toDensity(DataArray a,
                                      const std::vector<Dim> &dims);
SCIPP_CORE_EXPORT void fromDensity(const DataArrayView data,
                                   const std::vector<Variable> &binWidths);
SCIPP_CORE_EXPORT Dataset fromDensity(Dataset d, const Dim dim);
SCIPP_CORE_EXPORT Dataset fromDensity(Dataset d, const std::vector<Dim> &dims);
SCIPP_CORE_EXPORT DataArray fromDensity(DataArray a, const Dim dim);
SCIPP_CORE_EXPORT DataArray fromDensity(DataArray a,
                                        const std::vector<Dim> &dims);
} // namespace counts
} // namespace scipp::core

#endif // SCIPP_CORE_COUNTS_H
