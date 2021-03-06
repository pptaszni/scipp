// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <initializer_list>
#include <string>

#include "scipp-units_export.h"
#include "scipp/units/unit.h"

namespace scipp::units {

SCIPP_UNITS_EXPORT std::ostream &operator<<(std::ostream &os, const Dim dim);

SCIPP_UNITS_EXPORT std::string to_string(const units::Unit &unit);
template <class T> std::string to_string(const std::initializer_list<T> items);

} // namespace scipp::units
