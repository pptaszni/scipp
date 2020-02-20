// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock

#include "scipp/neutron/beamline.h"
#include "scipp/core/dataset.h"
#include "scipp/core/transform.h"

using namespace scipp::core;

namespace scipp::neutron {

namespace beamline_impl {

template <class T> static auto position(const T &d) {
  if (d.coords().contains(Dim::Position))
    return d.coords()[Dim::Position];
  else if (d.labels().contains("position"))
    return d.labels()["position"];
  else
    return d.attrs()["position"];
}

template <class T> static auto source_position(const T &d) {
  if (d.labels().contains("source_position"))
    return d.labels()["source_position"];
  else
    return d.attrs()["source_position"];
}

template <class T> static auto sample_position(const T &d) {
  if (d.labels().contains("sample_position"))
    return d.labels()["sample_position"];
  else
    return d.attrs()["sample_position"];
}

template <class T> static Variable flight_path_length(const T &d) {
  // If there is no sample this returns the straight distance from the source,
  // as required, e.g., for monitors.
  if (d.labels().contains("sample_position") ||
      d.attrs().contains("sample_position"))
    return l1(d) + l2(d);
  else
    return norm(position(d) - source_position(d));
}

template <class T> static Variable l1(const T &d) {
  return norm(sample_position(d) - source_position(d));
}

template <class T> static Variable l2(const T &d) {
  // Use transform to avoid temporaries. For certain unit conversions this can
  // cause a speedup >50%. Short version would be:
  //   return norm(position(d) - sample_position(d));
  return transform<pair_self_t<Eigen::Vector3d>>(
      position(d), sample_position(d),
      overloaded{
          [](const auto &x, const auto &y) { return (x - y).norm(); },
          [](const units::Unit &x, const units::Unit &y) { return x - y; }});
}

template <class T> static Variable scattering_angle(const T &d) {
  return 0.5 * two_theta(d);
}

template <class T> static Variable two_theta(const T &d) {
  auto beam = sample_position(d) - source_position(d);
  const auto l1 = norm(beam);
  beam /= l1;
  auto scattered = position(d) - sample_position(d);
  const auto l2 = norm(scattered);
  scattered /= l2;

  return acos(dot(beam, scattered));
}
} // namespace beamline_impl

core::VariableConstView position(const core::DatasetConstView &d) {
  return beamline_impl::position(d);
}
core::VariableConstView source_position(const core::DatasetConstView &d) {
  return beamline_impl::source_position(d);
}
core::VariableConstView sample_position(const core::DatasetConstView &d) {
  return beamline_impl::sample_position(d);
}
core::VariableView position(const core::DatasetView &d) {
  return beamline_impl::position(d);
}
core::VariableView source_position(const core::DatasetView &d) {
  return beamline_impl::source_position(d);
}
core::VariableView sample_position(const core::DatasetView &d) {
  return beamline_impl::sample_position(d);
}
core::Variable flight_path_length(const core::DatasetConstView &d) {
  return beamline_impl::flight_path_length(d);
}
core::Variable l1(const core::DatasetConstView &d) {
  return beamline_impl::l1(d);
}
core::Variable l2(const core::DatasetConstView &d) {
  return beamline_impl::l2(d);
}
core::Variable scattering_angle(const core::DatasetConstView &d) {
  return beamline_impl::scattering_angle(d);
}
core::Variable two_theta(const core::DatasetConstView &d) {
  return beamline_impl::two_theta(d);
}

core::VariableConstView position(const core::DataArrayConstView &d) {
  return beamline_impl::position(d);
}
core::VariableConstView source_position(const core::DataArrayConstView &d) {
  return beamline_impl::source_position(d);
}
core::VariableConstView sample_position(const core::DataArrayConstView &d) {
  return beamline_impl::sample_position(d);
}
core::VariableView position(const core::DataArrayView &d) {
  return beamline_impl::position(d);
}
core::VariableView source_position(const core::DataArrayView &d) {
  return beamline_impl::source_position(d);
}
core::VariableView sample_position(const core::DataArrayView &d) {
  return beamline_impl::sample_position(d);
}
core::Variable flight_path_length(const core::DataArrayConstView &d) {
  return beamline_impl::flight_path_length(d);
}
core::Variable l1(const core::DataArrayConstView &d) {
  return beamline_impl::l1(d);
}
core::Variable l2(const core::DataArrayConstView &d) {
  return beamline_impl::l2(d);
}
core::Variable scattering_angle(const core::DataArrayConstView &d) {
  return beamline_impl::scattering_angle(d);
}
core::Variable two_theta(const core::DataArrayConstView &d) {
  return beamline_impl::two_theta(d);
}

} // namespace scipp::neutron
