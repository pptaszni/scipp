// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "scipp/common/numeric.h"

using namespace scipp::numeric;

TEST(NumericTest, is_linspace_empty) {
  ASSERT_FALSE(is_linspace(std::vector<double>({})));
}

TEST(NumericTest, is_linspace_size_1) {
  ASSERT_FALSE(is_linspace(std::vector<double>({1.0})));
}

TEST(NumericTest, is_linspace_negative) {
  ASSERT_FALSE(is_linspace(std::vector<double>({1.0, 0.5})));
}

TEST(NumericTest, is_linspace_constant) {
  ASSERT_FALSE(is_linspace(std::vector<double>({1.0, 1.0, 1.0})));
}

TEST(NumericTest, is_linspace_constant_section) {
  ASSERT_FALSE(is_linspace(std::vector<double>({1.0, 1.0, 2.0})));
}

TEST(NumericTest, is_linspace_decreasing_section) {
  ASSERT_FALSE(is_linspace(std::vector<double>({1.5, 1.0, 2.0})));
}

TEST(NumericTest, is_linspace_size_2) {
  ASSERT_TRUE(is_linspace(std::vector<double>({1.0, 2.0})));
}

TEST(NumericTest, is_linspace_size_3) {
  ASSERT_TRUE(is_linspace(std::vector<double>({1.0, 2.0, 3.0})));
}

TEST(NumericTest, is_linspace_std_iota) {
  std::vector<double> range(1e5);
  std::iota(range.begin(), range.end(), 1e-9);
  ASSERT_TRUE(is_linspace(range));
}

TEST(NumericTest, is_linspace_generate_addition) {
  std::vector<double> range;
  double current = 345.4564675;
  std::generate_n(std::back_inserter(range), 1e5, [&current]() {
    const double step = 0.0034674;
    current += step;
    return current;
  });

  ASSERT_TRUE(is_linspace(range));
}

TEST(NumericTest, is_logspace_empty) {
  ASSERT_FALSE(is_logspace(std::vector<double>({})));
}

TEST(NumericTest, is_logspace_size_1) {
  ASSERT_FALSE(is_logspace(std::vector<double>({1.0})));
}

TEST(NumericTest, is_logspace_negative) {
  ASSERT_FALSE(is_logspace(std::vector<double>({1.0, 0.5})));
}

TEST(NumericTest, is_logspace_constant) {
  ASSERT_FALSE(is_logspace(std::vector<double>({1.0, 1.0, 1.0})));
}

TEST(NumericTest, is_logspace_constant_section) {
  ASSERT_FALSE(is_logspace(std::vector<double>({1.0, 20.0, 20.0})));
}

TEST(NumericTest, is_logspace_decreasing_section) {
  ASSERT_FALSE(is_logspace(std::vector<double>({1.5, 1.0, 2.0})));
}

TEST(NumericTest, is_logspace_ratio_2) {
  ASSERT_TRUE(is_logspace(std::vector<double>({1, 2, 4, 8, 16, 32})));
}

TEST(NumericTest, is_logspace_ratio_3) {
  ASSERT_TRUE(is_logspace(std::vector<double>({1, 3, 9, 27, 81, 243})));
}

TEST(NumericTest, is_logspace_ratio_4point8) {
  ASSERT_TRUE(
      is_logspace(std::vector<double>({4, 4.8, 5.76, 6.912, 8.2944, 9.95328})));
}

TEST(NumericTest, is_logspace_ratio_no_quite_4point8) {
  ASSERT_FALSE(is_logspace(
      std::vector<double>({4, 4.8, 5.76, 6.912, 8.2944, 9.953280000001})));
}

TEST(NumericTest, is_logspace_ratio_100) {
  ASSERT_TRUE(is_logspace(
      std::vector<double>({1, 1.0e2, 1.0e4, 1.0e6, 1.0e8, 1.0e10})));
}

TEST(NumericTest, is_logspace_negative_numbers) {
  ASSERT_TRUE(is_logspace(std::vector<double>({-32, -16, -8, -4, -2, -1})));
}

TEST(NumericTest, is_logspace_exit_if_front_is_zero) {
  ASSERT_FALSE(is_logspace(std::vector<double>({0, 1, 2, 3})));
}
