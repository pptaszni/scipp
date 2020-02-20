// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <cmath>

#include "scipp/core/transform.h"
#include "scipp/core/variable.h"

namespace scipp::core {

static constexpr auto plus_ = [](const auto a_, const auto b_) {
  return a_ + b_;
};
static constexpr auto minus_ = [](const auto a_, const auto b_) {
  return a_ - b_;
};
static constexpr auto times_ = [](const auto a_, const auto b_) {
  return a_ * b_;
};
static constexpr auto divide_ = [](const auto a_, const auto b_) {
  return a_ / b_;
};

template <class T1, class T2> Variable plus(const T1 &a, const T2 &b) {
  return transform<arithmetic_and_matrix_type_pairs>(a, b, plus_);
}

Variable Variable::operator-() const {
  return transform<double, float, int64_t, Eigen::Vector3d>(
      *this, [](const auto a) { return -a; });
}

template <class T1, class T2> Variable minus(const T1 &a, const T2 &b) {
  return transform<arithmetic_and_matrix_type_pairs>(a, b, minus_);
}

template <class T1, class T2> Variable times(const T1 &a, const T2 &b) {
  return transform<arithmetic_type_pairs_with_bool>(a, b, times_);
}

template <class T1, class T2> Variable divide(const T1 &a, const T2 &b) {
  return transform<arithmetic_type_pairs>(a, b, divide_);
}

Variable VariableConstView::operator-() const {
  Variable copy(*this);
  return -copy;
}

Variable operator+(const VariableConstView &a, const VariableConstView &b) {
  return plus(a, b);
}
Variable operator-(const VariableConstView &a, const VariableConstView &b) {
  return minus(a, b);
}
Variable operator*(const VariableConstView &a, const VariableConstView &b) {
  return times(a, b);
}
Variable operator/(const VariableConstView &a, const VariableConstView &b) {
  return divide(a, b);
}
Variable operator+(const VariableConstView &a_, const double b) {
  Variable a(a_);
  return a += b;
}
Variable operator-(const VariableConstView &a_, const double b) {
  Variable a(a_);
  return a -= b;
}
Variable operator*(const VariableConstView &a_, const double b) {
  Variable a(a_);
  return a *= b;
}
Variable operator/(const VariableConstView &a_, const double b) {
  Variable a(a_);
  return a /= b;
}
Variable operator+(const double a, const VariableConstView &b_) {
  Variable b(b_);
  return b += a;
}
Variable operator-(const double a, const VariableConstView &b_) {
  Variable b(b_);
  return -(b -= a);
}
Variable operator*(const double a, const VariableConstView &b_) {
  Variable b(b_);
  return b *= a;
}
Variable operator/(const double a, const VariableConstView &b_view) {
  Variable b(b_view);
  transform_in_place<double, float>(
      b,
      overloaded{
          [](units::Unit &b_) { b_ = units::Unit(units::dimensionless) / b_; },
          [a](double &b_) { b_ = a / b_; },
          [a](float &b_) { b_ = static_cast<float>(a / b_); },
          [a](auto &b_) {
            b_ = static_cast<std::remove_reference_t<decltype(b_)>>(a / b_);
          }});
  return b;
}

} // namespace scipp::core
