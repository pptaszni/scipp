// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>
#include <vector>

#include "test_macros.h"

#include "fix_typed_test_suite_warnings.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/except.h"
#include "scipp/core/variable.h"

#include "../element_unary_operations.h"

using namespace scipp;
using namespace scipp::core;

template <typename T>
class VariableScalarOperatorTest : public ::testing::Test {
public:
  Variable variable{makeVariable<T>(Dims{Dim::X}, Shape{1}, Values{10})};
  const T scalar{2};

  T value() const { return this->variable.template values<T>()[0]; }
};

using ScalarTypes = ::testing::Types<double, float, int64_t, int32_t>;
TYPED_TEST_SUITE(VariableScalarOperatorTest, ScalarTypes);

TYPED_TEST(VariableScalarOperatorTest, plus_equals) {
  this->variable += this->scalar;
  EXPECT_EQ(this->value(), 12);
}

TYPED_TEST(VariableScalarOperatorTest, minus_equals) {
  this->variable -= this->scalar;
  EXPECT_EQ(this->value(), 8);
}

TYPED_TEST(VariableScalarOperatorTest, times_equals) {
  this->variable *= this->scalar;
  EXPECT_EQ(this->value(), 20);
}

TYPED_TEST(VariableScalarOperatorTest, divide_equals) {
  this->variable /= this->scalar;
  EXPECT_EQ(this->value(), 5);
}

TEST(Variable, operator_unary_minus) {
  const auto a = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                      units::Unit(units::m), Values{1.1, 2.2});
  const auto expected = makeVariable<double>(
      Dims{Dim::X}, Shape{2}, units::Unit(units::m), Values{-1.1, -2.2});
  auto b = -a;
  EXPECT_EQ(b, expected);
}

TEST(VariableView, unary_minus) {
  const auto a = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                      units::Unit(units::m), Values{1.1, 2.2});
  const auto expected = makeVariable<double>(
      Dims(), Shape(), units::Unit(units::m), Values{-2.2});
  auto b = -a.slice({Dim::X, 1});
  EXPECT_EQ(b, expected);
}

TEST(Variable, operator_plus_equal) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  ASSERT_NO_THROW(a += a);
  EXPECT_EQ(a.values<double>()[0], 2.2);
  EXPECT_EQ(a.values<double>()[1], 4.4);
}

TEST(Variable, operator_plus_equal_automatic_broadcast_of_rhs) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  auto fewer_dimensions = makeVariable<double>(Values{1.0});

  ASSERT_NO_THROW(a += fewer_dimensions);
  EXPECT_EQ(a.values<double>()[0], 2.1);
  EXPECT_EQ(a.values<double>()[1], 3.2);
}

TEST(Variable, operator_plus_equal_transpose) {
  auto a = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 2},
                                units::Unit(units::m),
                                Values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  auto transpose = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 3},
                                        units::Unit(units::m),
                                        Values{1.0, 3.0, 5.0, 2.0, 4.0, 6.0});

  EXPECT_NO_THROW(a += transpose);
  ASSERT_EQ(a, makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 2},
                                    units::Unit(units::m),
                                    Values{2.0, 4.0, 6.0, 8.0, 10.0, 12.0}));
}

TEST(Variable, operator_plus_equal_different_dimensions) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  auto different_dimensions =
      makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{1.1, 2.2});
  EXPECT_THROW_MSG(a += different_dimensions, std::runtime_error,
                   "Expected {{x, 2}} to contain {{y, 2}}.");
}

TEST(Variable, operator_plus_equal_different_unit) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  auto different_unit(a);
  different_unit.setUnit(units::m);
  EXPECT_THROW(a += different_unit, except::UnitError);
}

TEST(Variable, operator_plus_equal_non_arithmetic_type) {
  auto a = makeVariable<std::string>(Dims{Dim::X}, Shape{1},
                                     Values{std::string("test")});
  EXPECT_THROW(a += a, except::TypeError);
}

TEST(Variable, operator_plus_equal_different_variables_different_element_type) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  auto b = makeVariable<int64_t>(Dims{Dim::X}, Shape{1}, Values{2});
  EXPECT_THROW(a += b, except::TypeError);
}

TEST(Variable, operator_plus_equal_different_variables_same_element_type) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  auto b = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{2.0});
  EXPECT_NO_THROW(a += b);
  EXPECT_EQ(a.values<double>()[0], 3.0);
}

TEST(Variable, operator_plus_equal_scalar) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  EXPECT_NO_THROW(a += 1.0);
  EXPECT_EQ(a.values<double>()[0], 2.1);
  EXPECT_EQ(a.values<double>()[1], 3.2);
}

TEST(Variable, operator_plus_equal_custom_type) {
  auto a = makeVariable<float>(Dims{Dim::X}, Shape{2}, Values{1.1f, 2.2f});

  EXPECT_NO_THROW(a += a);
  EXPECT_EQ(a.values<float>()[0], 2.2f);
  EXPECT_EQ(a.values<float>()[1], 4.4f);
}

TEST(Variable, operator_plus) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 2.0},
                                Variances{3.0, 4.0});
  auto b =
      makeVariable<float>(Dims{Dim::Y, Dim::Z}, Shape{2, Dimensions::Sparse});
  auto b_ = b.sparseValues<float>();
  b_[0] = {0.1, 0.2};
  b_[1] = {0.3};

  auto sum = a + b;

  auto expected = makeVariable<double>(
      Dimensions{{Dim::X, 2}, {Dim::Y, 2}, {Dim::Z, Dimensions::Sparse}},
      Variances{}, Values{});
  auto vals = expected.sparseValues<double>();
  vals[0] = {1.0 + 0.1f, 1.0 + 0.2f};
  vals[1] = {1.0 + 0.3f};
  vals[2] = {2.0 + 0.1f, 2.0 + 0.2f};
  vals[3] = {2.0 + 0.3f};
  auto vars = expected.sparseVariances<double>();
  vars[0] = {3.0, 3.0};
  vars[1] = {3.0};
  vars[2] = {4.0, 4.0};
  vars[3] = {4.0};
  EXPECT_EQ(sum, expected);
}

TEST(Variable, operator_plus_unit_fail) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 2.0},
                                Variances{3.0, 4.0});
  a.setUnit(units::m);
  auto b = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 2.0},
                                Variances{3.0, 4.0});
  b.setUnit(units::s);
  ASSERT_ANY_THROW(a + b);
  b.setUnit(units::m);
  ASSERT_NO_THROW(a + b);
}

TEST(Variable, operator_plus_eigen_type) {
  const auto var = makeVariable<Eigen::Vector3d>(
      Dims{Dim::X}, Shape{2},
      Values{Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Vector3d{0.1, 0.2, 0.3}});
  const auto expected = makeVariable<Eigen::Vector3d>(
      Dims(), Shape(), Values{Eigen::Vector3d{1.1, 2.2, 3.3}});

  const auto result = var.slice({Dim::X, 0}) + var.slice({Dim::X, 1});

  EXPECT_EQ(result, expected);
}

TEST(SparseVariable, operator_plus) {
  auto sparse =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2l, Dimensions::Sparse});
  auto sparse_ = sparse.sparseValues<double>();
  sparse_[0] = {1, 2, 3};
  sparse_[1] = {4};
  auto dense = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{1.5, 0.5});

  sparse += dense;

  EXPECT_TRUE(equals(sparse_[0], {2.5, 3.5, 4.5}));
  EXPECT_TRUE(equals(sparse_[1], {4.5}));
}

TEST(Variable, operator_times_equal) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::m),
                                Values{2.0, 3.0});

  EXPECT_EQ(a.unit(), units::m);
  EXPECT_NO_THROW(a *= a);
  EXPECT_EQ(a.values<double>()[0], 4.0);
  EXPECT_EQ(a.values<double>()[1], 9.0);
  EXPECT_EQ(a.unit(), units::m * units::m);
}

TEST(Variable, operator_times_equal_scalar) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::m),
                                Values{2.0, 3.0});

  EXPECT_EQ(a.unit(), units::m);
  EXPECT_NO_THROW(a *= 2.0);
  EXPECT_EQ(a.values<double>()[0], 4.0);
  EXPECT_EQ(a.values<double>()[1], 6.0);
  EXPECT_EQ(a.unit(), units::m);
}

TEST(Variable, operator_times_equal_unit_fail_integrity) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2},
                           units::Unit(units::m * units::m), Values{2.0, 3.0});
  const auto expected(a);

  // This test relies on m^4 being an unsupported unit.
  ASSERT_THROW(a *= a, std::runtime_error);
  EXPECT_EQ(a, expected);
}

TEST(Variable, operator_binary_equal_data_fail_unit_integrity) {
  auto a =
      makeVariable<float>(Dims{Dim::Y, Dim::Z}, Shape{2, Dimensions::Sparse});
  auto a_ = a.sparseValues<float>();
  auto b(a);
  a_[0] = {0.1, 0.2};
  a_[1] = {0.3};
  b.setUnit(units::m);
  auto expected(a);

  ASSERT_THROW(a *= b, except::SizeError);
  EXPECT_EQ(a, expected);
  ASSERT_THROW(a /= b, except::SizeError);
  EXPECT_EQ(a, expected);
}

TEST(Variable, operator_binary_equal_data_fail_data_integrity) {
  auto a =
      makeVariable<float>(Dims{Dim::Y, Dim::Z}, Shape{2, Dimensions::Sparse});
  auto a_ = a.sparseValues<float>();
  a_[0] = {0.1, 0.2};
  auto b(a);
  a_[1] = {0.3};
  b.setUnit(units::m);
  auto expected(a);

  ASSERT_THROW(a *= b, except::SizeError);
  EXPECT_EQ(a, expected);
  ASSERT_THROW(a /= b, except::SizeError);
  EXPECT_EQ(a, expected);
}

TEST(Variable, operator_binary_equal_with_variances_data_fail_data_integrity) {
  auto a =
      makeVariable<float>(Dimensions{{Dim::Y, 2}, {Dim::Z, Dimensions::Sparse}},
                          Values{}, Variances{});
  auto a_ = a.sparseValues<float>();
  auto a_vars = a.sparseVariances<float>();
  a_[0] = {0.1, 0.2};
  a_vars[0] = {0.1, 0.2};
  auto b(a);
  a_[1] = {0.3};
  a_vars[1] = {0.3};
  b.setUnit(units::m);
  auto expected(a);

  // Length mismatch of second sparse item
  ASSERT_THROW(a *= b, except::SizeError);
  EXPECT_EQ(a, expected);
  ASSERT_THROW(a /= b, except::SizeError);
  EXPECT_EQ(a, expected);

  b = a;
  b.setUnit(units::m);
  a_vars[1].clear();
  expected = a;

  // Length mismatch between values and variances
  ASSERT_THROW(a *= b, except::SizeError);
  EXPECT_EQ(a, expected);
  ASSERT_THROW(a /= b, except::SizeError);
  EXPECT_EQ(a, expected);
}

TEST(Variable, operator_times_equal_slice_unit_fail_integrity) {
  auto a =
      makeVariable<float>(Dims{Dim::Y, Dim::Z}, Shape{2, Dimensions::Sparse});
  auto a_ = a.sparseValues<float>();
  a_[0] = {0.1, 0.2};
  a_[1] = {0.3};
  auto b(a);
  b.setUnit(units::m);
  auto expected(a);

  ASSERT_THROW(a.slice({Dim::Y, 0}) *= b.slice({Dim::Y, 0}), except::UnitError);
  EXPECT_EQ(a, expected);
}

TEST(Variable, operator_times_can_broadcast) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{0.5, 1.5});
  auto b = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{2.0, 3.0});

  auto ab = a * b;
  auto reference = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                        Values{1.0, 3.0, 1.5, 4.5});
  EXPECT_EQ(ab, reference);
}

TEST(Variable, operator_divide_equal) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0, 3.0});
  auto b = makeVariable<double>(Values{2.0});
  b.setUnit(units::m);

  EXPECT_NO_THROW(a /= b);
  EXPECT_EQ(a.values<double>()[0], 1.0);
  EXPECT_EQ(a.values<double>()[1], 1.5);
  EXPECT_EQ(a.unit(), units::dimensionless / units::m);
}

TEST(Variable, operator_divide_equal_self) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::m),
                                Values{2.0, 3.0});

  EXPECT_EQ(a.unit(), units::m);
  EXPECT_NO_THROW(a /= a);
  EXPECT_EQ(a.values<double>()[0], 1.0);
  EXPECT_EQ(a.values<double>()[1], 1.0);
  EXPECT_EQ(a.unit(), units::dimensionless);
}

TEST(Variable, operator_divide_equal_scalar) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::m),
                                Values{2.0, 4.0});

  EXPECT_EQ(a.unit(), units::m);
  EXPECT_NO_THROW(a /= 2.0);
  EXPECT_EQ(a.values<double>()[0], 1.0);
  EXPECT_EQ(a.values<double>()[1], 2.0);
  EXPECT_EQ(a.unit(), units::m);
}

TEST(Variable, operator_divide_scalar_double) {
  const auto a = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                      units::Unit(units::m), Values{2.0, 4.0});
  const auto result = 1.111 / a;
  EXPECT_EQ(result.values<double>()[0], 1.111 / 2.0);
  EXPECT_EQ(result.values<double>()[1], 1.111 / 4.0);
  EXPECT_EQ(result.unit(), units::dimensionless / units::m);
}

TEST(Variable, operator_divide_scalar_float) {
  const auto a = makeVariable<float>(Dims{Dim::X}, Shape{2},
                                     units::Unit(units::m), Values{2.0, 4.0});
  const auto result = 1.111f / a;
  EXPECT_EQ(result.values<float>()[0], 1.111f / 2.0f);
  EXPECT_EQ(result.values<float>()[1], 1.111f / 4.0f);
  EXPECT_EQ(result.unit(), units::dimensionless / units::m);
}

TEST(Variable, operator_allowed_types) {
  auto i32 = makeVariable<int32_t>(Values{10});
  auto i64 = makeVariable<int64_t>(Values{10});
  auto f = makeVariable<float>(Values{0.5f});
  auto d = makeVariable<double>(Values{0.5});

  /* Can operate on higher precision from lower precision */
  EXPECT_NO_THROW(i64 += i32);
  EXPECT_NO_THROW(d += f);

  /* Can not operate on lower precision from higher precision */
  EXPECT_ANY_THROW(i32 += i64);
  EXPECT_ANY_THROW(f += d);

  /* Expect promotion to double if one parameter is double */
  EXPECT_EQ(dtype<double>, (f + d).dtype());
  EXPECT_EQ(dtype<double>, (d + f).dtype());
}

TEST(Variable, concatenate) {
  Dimensions dims(Dim::Tof, 1);
  auto a = makeVariable<double>(Dimensions(dims), Values{1.0});
  auto b = makeVariable<double>(Dimensions(dims), Values{2.0});
  a.setUnit(units::m);
  b.setUnit(units::m);
  auto ab = concatenate(a, b, Dim::Tof);
  ASSERT_EQ(ab.dims().volume(), 2);
  EXPECT_EQ(ab.unit(), units::Unit(units::m));
  const auto &data = ab.values<double>();
  EXPECT_EQ(data[0], 1.0);
  EXPECT_EQ(data[1], 2.0);
  auto ba = concatenate(b, a, Dim::Tof);
  const auto abba = concatenate(ab, ba, Dim::Q);
  ASSERT_EQ(abba.dims().volume(), 4);
  EXPECT_EQ(abba.dims().shape().size(), 2);
  const auto &data2 = abba.values<double>();
  EXPECT_EQ(data2[0], 1.0);
  EXPECT_EQ(data2[1], 2.0);
  EXPECT_EQ(data2[2], 2.0);
  EXPECT_EQ(data2[3], 1.0);
  const auto ababbaba = concatenate(abba, abba, Dim::Tof);
  ASSERT_EQ(ababbaba.dims().volume(), 8);
  const auto &data3 = ababbaba.values<double>();
  EXPECT_EQ(data3[0], 1.0);
  EXPECT_EQ(data3[1], 2.0);
  EXPECT_EQ(data3[2], 1.0);
  EXPECT_EQ(data3[3], 2.0);
  EXPECT_EQ(data3[4], 2.0);
  EXPECT_EQ(data3[5], 1.0);
  EXPECT_EQ(data3[6], 2.0);
  EXPECT_EQ(data3[7], 1.0);
  const auto abbaabba = concatenate(abba, abba, Dim::Q);
  ASSERT_EQ(abbaabba.dims().volume(), 8);
  const auto &data4 = abbaabba.values<double>();
  EXPECT_EQ(data4[0], 1.0);
  EXPECT_EQ(data4[1], 2.0);
  EXPECT_EQ(data4[2], 2.0);
  EXPECT_EQ(data4[3], 1.0);
  EXPECT_EQ(data4[4], 1.0);
  EXPECT_EQ(data4[5], 2.0);
  EXPECT_EQ(data4[6], 2.0);
  EXPECT_EQ(data4[7], 1.0);
}

TEST(Variable, concatenate_volume_with_slice) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  auto aa = concatenate(a, a, Dim::X);
  EXPECT_NO_THROW(concatenate(aa, a, Dim::X));
}

TEST(Variable, concatenate_slice_with_volume) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  auto aa = concatenate(a, a, Dim::X);
  EXPECT_NO_THROW(concatenate(a, aa, Dim::X));
}

TEST(Variable, concatenate_fail) {
  Dimensions dims(Dim::Tof, 1);
  auto a = makeVariable<double>(Dimensions(dims), Values{1.0});
  auto b = makeVariable<double>(Dimensions(dims), Values{2.0});
  auto c = makeVariable<float>(Dimensions(dims), Values{2.0});
  EXPECT_THROW_MSG(concatenate(a, c, Dim::Tof), std::runtime_error,
                   "Cannot concatenate Variables: Data types do not match.");
  auto aa = concatenate(a, a, Dim::Tof);
  EXPECT_THROW_MSG(
      concatenate(a, aa, Dim::Q), std::runtime_error,
      "Cannot concatenate Variables: Dimension extents do not match.");
}

TEST(Variable, concatenate_unit_fail) {
  Dimensions dims(Dim::X, 1);
  auto a = makeVariable<double>(Dimensions(dims), Values{1.0});
  auto b(a);
  EXPECT_NO_THROW(concatenate(a, b, Dim::X));
  a.setUnit(units::m);
  EXPECT_THROW_MSG(concatenate(a, b, Dim::X), std::runtime_error,
                   "Cannot concatenate Variables: Units do not match.");
  b.setUnit(units::m);
  EXPECT_NO_THROW(concatenate(a, b, Dim::X));
}

TEST(SparseVariable, concatenate) {
  const auto a =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, Dimensions::Sparse},
                           Values{}, Variances{});
  const auto b =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, Dimensions::Sparse},
                           Values{}, Variances{});
  auto var = concatenate(a, b, Dim::Y);
  EXPECT_EQ(var, makeVariable<double>(Dims{Dim::Y, Dim::X},
                                      Shape{5, Dimensions::Sparse}, Values{},
                                      Variances{}));
}

TEST(SparseVariable, concatenate_along_sparse_dimension) {
  auto a =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2l, Dimensions::Sparse});
  auto a_ = a.sparseValues<double>();
  a_[0] = {1, 2, 3};
  a_[1] = {1, 2};
  auto b =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2l, Dimensions::Sparse});
  auto b_ = b.sparseValues<double>();
  b_[0] = {1, 3};
  b_[1] = {};

  auto var = concatenate(a, b, Dim::X);
  EXPECT_TRUE(var.dims().sparse());
  EXPECT_EQ(var.dims().sparseDim(), Dim::X);
  EXPECT_EQ(var.dims().volume(), 2);
  auto data = var.sparseValues<double>();
  EXPECT_TRUE(equals(data[0], {1, 2, 3, 1, 3}));
  EXPECT_TRUE(equals(data[1], {1, 2}));
}

TEST(SparseVariable, concatenate_along_sparse_dimension_with_variances) {
  auto a =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, Dimensions::Sparse},
                           Values{}, Variances{});
  auto a_vals = a.sparseValues<double>();
  a_vals[0] = {1, 2, 3};
  a_vals[1] = {1, 2};
  auto a_vars = a.sparseVariances<double>();
  a_vars[0] = {4, 5, 6};
  a_vars[1] = {4, 5};
  auto b =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, Dimensions::Sparse},
                           Values{}, Variances{});
  auto b_vals = b.sparseValues<double>();
  b_vals[0] = {1, 3};
  b_vals[1] = {};
  auto b_vars = b.sparseVariances<double>();
  b_vars[0] = {7, 8};
  b_vars[1] = {};

  auto var = concatenate(a, b, Dim::X);
  EXPECT_TRUE(var.dims().sparse());
  EXPECT_EQ(var.dims().sparseDim(), Dim::X);
  EXPECT_EQ(var.dims().volume(), 2);
  auto vals = var.sparseValues<double>();
  EXPECT_TRUE(equals(vals[0], {1, 2, 3, 1, 3}));
  EXPECT_TRUE(equals(vals[1], {1, 2}));
  auto vars = var.sparseVariances<double>();
  EXPECT_TRUE(equals(vars[0], {4, 5, 6, 7, 8}));
  EXPECT_TRUE(equals(vars[1], {4, 5}));
}

#ifdef SCIPP_UNITS_NEUTRON
TEST(Variable, rebin) {
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 2.0});
  var.setUnit(units::counts);
  const auto oldEdge =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1.0, 2.0, 3.0});
  const auto newEdge =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 3.0});
  auto rebinned = rebin(var, Dim::X, oldEdge, newEdge);
  ASSERT_EQ(rebinned.dims().shape().size(), 1);
  ASSERT_EQ(rebinned.dims().volume(), 1);
  ASSERT_EQ(rebinned.values<double>().size(), 1);
  EXPECT_EQ(rebinned.values<double>()[0], 3.0);
}
#endif

TEST(Variable, sum) {
  const auto var =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                           units::Unit(units::m), Values{1.0, 2.0, 3.0, 4.0});
  const auto expectedX = makeVariable<double>(
      Dims{Dim::Y}, Shape{2}, units::Unit(units::m), Values{3.0, 7.0});
  const auto expectedY = makeVariable<double>(
      Dims{Dim::X}, Shape{2}, units::Unit(units::m), Values{4.0, 6.0});
  EXPECT_EQ(sum(var, Dim::X), expectedX);
  EXPECT_EQ(sum(var, Dim::Y), expectedY);
}

TEST(Variable, sum_in_place) {
  const auto var =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                           units::Unit(units::m), Values{1.0, 2.0, 3.0, 4.0});

  auto out =
      makeVariable<double>(Dims{Dim::Y}, Shape{2}, units::Unit{units::m});
  auto view = sum(var, Dim::X, out);

  const auto expected = makeVariable<double>(
      Dims{Dim::Y}, Shape{2}, units::Unit(units::m), Values{3.0, 7.0});

  EXPECT_EQ(out, expected);
  EXPECT_EQ(view, out);
  EXPECT_EQ(view.underlying(), out);
}

TEST(Variable, sum_in_place_bool) {
  const auto var = makeVariable<bool>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                      Values{true, false, true, true});

  auto out = makeVariable<int64_t>(Dims{Dim::Y}, Shape{2});
  auto view = sum(var, Dim::X, out);

  const auto expected =
      makeVariable<int64_t>(Dims{Dim::Y}, Shape{2}, Values{1, 2});

  EXPECT_EQ(out, expected);
  EXPECT_EQ(view, out);
  EXPECT_EQ(view.underlying(), out);
}

TEST(Variable, sum_in_place_bool_incorrect_out_type) {
  const auto var = makeVariable<bool>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                      Values{true, false, true, true});
  auto out = makeVariable<float>(Dims{Dim::Y}, Shape{2});
  EXPECT_THROW(sum(var, Dim::X, out), except::UnitError);
}

TEST(VariableConstView, sum) {
  const auto var =
      makeVariable<float>(Dims{Dim::X}, Shape{4}, Values{1.0, 2.0, 3.0, 4.0});
  EXPECT_EQ(sum(var.slice({Dim::X, 0, 2}), Dim::X),
            makeVariable<float>(Values{3}));
}

TEST(Variable, abs) {
  auto reference =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                           units::Unit(units::m), Values{1, 2, 3, 4});
  auto var =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                           units::Unit(units::m), Values{1.0, -2.0, -3.0, 4.0});
  EXPECT_EQ(abs(var), reference);
}

TEST(Variable, norm_of_vector) {
  auto reference =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, units::Unit(units::m),
                           Values{sqrt(2.0), sqrt(2.0), 2.0});
  auto var = makeVariable<Eigen::Vector3d>(
      Dims{Dim::X}, Shape{3}, units::Unit(units::m),
      Values{Eigen::Vector3d{1, 0, -1}, Eigen::Vector3d{1, 1, 0},
             Eigen::Vector3d{0, 0, -2}});
  EXPECT_EQ(norm(var), reference);
}

TEST(Variable, sqrt) {
  const auto f64 = makeVariable<double>(Values{1.23});
  EXPECT_EQ(sqrt(f64), makeVariable<double>(Values{element::sqrt(1.23)}));
  const auto f32 = makeVariable<float>(Values{1.23456789});
  EXPECT_EQ(sqrt(f32), makeVariable<float>(Values{element::sqrt(1.23456789f)}));
}

TEST(Variable, sqrt_move) {
  auto var = makeVariable<double>(Values{1.23});
  const auto ptr = var.values<double>().data();
  auto out = sqrt(std::move(var));
  EXPECT_EQ(out, makeVariable<double>(Values{element::sqrt(1.23)}));
  EXPECT_EQ(out.values<double>().data(), ptr);
}

TEST(Variable, sqrt_out_arg) {
  auto x = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.23, 0.0});
  auto out = x.slice({Dim::X, 1});
  auto view = sqrt(x.slice({Dim::X, 0}), out);

  EXPECT_EQ(x, makeVariable<double>(Dims{Dim::X}, Shape{2},
                                    Values{1.23, element::sqrt(1.23)}));
  EXPECT_EQ(view, out);
  EXPECT_EQ(view.underlying(), x);
}

TEST(VariableReciprocalOutArg, full_in_place) {
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{3}, units::Unit(units::m),
                                  Values{1, 4, 9});
  auto view = reciprocal(var, var);
  EXPECT_EQ(var,
            makeVariable<double>(Dims{Dim::X}, Shape{3},
                                 units::Unit(units::dimensionless / units::m),
                                 Values{1., 1. / 4., 1. / 9.}));
  EXPECT_EQ(view, var);
  EXPECT_EQ(view.underlying(), var);
}

TEST(VariableReciprocalOutArg, partial) {
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3},
                                        units::Unit(units::m), Values{1, 4, 9});
  auto out =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::m));
  auto view = reciprocal(var.slice({Dim::X, 1, 3}), out);
  EXPECT_EQ(out,
            makeVariable<double>(Dims{Dim::X}, Shape{2},
                                 units::Unit(units::dimensionless / units::m),
                                 Values{1. / 4., 1. / 9.}));
  EXPECT_EQ(view, out);
  EXPECT_EQ(view.underlying(), out);
}

TEST(VariableAbsOutArg, full_in_place) {
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{3}, units::Unit(units::m),
                                  Values{1, -4, -9});
  auto view = abs(var, var);
  EXPECT_EQ(var, makeVariable<double>(Dims{Dim::X}, Shape{3},
                                      units::Unit(units::m), Values{1, 4, 9}));
  EXPECT_EQ(view, var);
  EXPECT_EQ(view.underlying(), var);
}

TEST(VariableAbsOutArg, partial) {
  const auto var = makeVariable<double>(
      Dims{Dim::X}, Shape{3}, units::Unit(units::m), Values{1, -4, -9});
  auto out =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::m));
  auto view = abs(var.slice({Dim::X, 1, 3}), out);
  EXPECT_EQ(out, makeVariable<double>(Dims{Dim::X}, Shape{2},
                                      units::Unit(units::m), Values{4, 9}));
  EXPECT_EQ(view, out);
  EXPECT_EQ(view.underlying(), out);
}

TEST(VariableView, minus_equals_failures) {
  auto var = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});

  EXPECT_THROW_MSG(var -= var.slice({Dim::X, 0, 1}), std::runtime_error,
                   "Expected {{x, 2}, {y, 2}} to contain {{x, 1}, {y, 2}}.");
}

TEST(VariableView, self_overlapping_view_operation) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});

  var -= var.slice({Dim::Y, 0});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  // This is the critical part: After subtracting for y=0 the view points to
  // data containing 0.0, so subsequently the subtraction would have no effect
  // if self-overlap was not taken into account by the implementation.
  EXPECT_EQ(data[2], 2.0);
  EXPECT_EQ(data[3], 2.0);
}

TEST(VariableView, minus_equals_slice_const_outer) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  const auto copy(var);

  var -= copy.slice({Dim::Y, 0});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  EXPECT_EQ(data[2], 2.0);
  EXPECT_EQ(data[3], 2.0);
  var -= copy.slice({Dim::Y, 1});
  EXPECT_EQ(data[0], -3.0);
  EXPECT_EQ(data[1], -4.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], -2.0);
}

TEST(VariableView, minus_equals_slice_outer) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  auto copy(var);

  var -= copy.slice({Dim::Y, 0});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  EXPECT_EQ(data[2], 2.0);
  EXPECT_EQ(data[3], 2.0);
  var -= copy.slice({Dim::Y, 1});
  EXPECT_EQ(data[0], -3.0);
  EXPECT_EQ(data[1], -4.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], -2.0);
}

TEST(VariableView, minus_equals_slice_inner) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  auto copy(var);

  var -= copy.slice({Dim::X, 0});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 1.0);
  EXPECT_EQ(data[2], 0.0);
  EXPECT_EQ(data[3], 1.0);
  var -= copy.slice({Dim::X, 1});
  EXPECT_EQ(data[0], -2.0);
  EXPECT_EQ(data[1], -1.0);
  EXPECT_EQ(data[2], -4.0);
  EXPECT_EQ(data[3], -3.0);
}

TEST(VariableView, minus_equals_slice_of_slice) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  auto copy(var);

  var -= copy.slice({Dim::X, 1}).slice({Dim::Y, 1});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], -3.0);
  EXPECT_EQ(data[1], -2.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], 0.0);
}

TEST(VariableView, minus_equals_nontrivial_slices) {
  auto source = makeVariable<double>(
      Dims{Dim::Y, Dim::X}, Shape{3, 3},
      Values{11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0});
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
    target -= source.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -11.0);
    EXPECT_EQ(data[1], -12.0);
    EXPECT_EQ(data[2], -21.0);
    EXPECT_EQ(data[3], -22.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
    target -= source.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -12.0);
    EXPECT_EQ(data[1], -13.0);
    EXPECT_EQ(data[2], -22.0);
    EXPECT_EQ(data[3], -23.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
    target -= source.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -21.0);
    EXPECT_EQ(data[1], -22.0);
    EXPECT_EQ(data[2], -31.0);
    EXPECT_EQ(data[3], -32.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
    target -= source.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -22.0);
    EXPECT_EQ(data[1], -23.0);
    EXPECT_EQ(data[2], -32.0);
    EXPECT_EQ(data[3], -33.0);
  }
}

TEST(VariableView, slice_inner_minus_equals) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});

  var.slice({Dim::X, 0}) -= var.slice({Dim::X, 1});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], -1.0);
  EXPECT_EQ(data[1], 2.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], 4.0);
}

TEST(VariableView, slice_outer_minus_equals) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});

  var.slice({Dim::Y, 0}) -= var.slice({Dim::Y, 1});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], -2.0);
  EXPECT_EQ(data[1], -2.0);
  EXPECT_EQ(data[2], 3.0);
  EXPECT_EQ(data[3], 4.0);
}

TEST(VariableView, nontrivial_slice_minus_equals) {
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                       Values{11.0, 12.0, 21.0, 22.0});
    target.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2}) -= source;
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -11.0);
    EXPECT_EQ(data[1], -12.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], -21.0);
    EXPECT_EQ(data[4], -22.0);
    EXPECT_EQ(data[5], 0.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 0.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                       Values{11.0, 12.0, 21.0, 22.0});
    target.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2}) -= source;
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], -11.0);
    EXPECT_EQ(data[2], -12.0);
    EXPECT_EQ(data[3], 0.0);
    EXPECT_EQ(data[4], -21.0);
    EXPECT_EQ(data[5], -22.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 0.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                       Values{11.0, 12.0, 21.0, 22.0});
    target.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3}) -= source;
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 0.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], -11.0);
    EXPECT_EQ(data[4], -12.0);
    EXPECT_EQ(data[5], 0.0);
    EXPECT_EQ(data[6], -21.0);
    EXPECT_EQ(data[7], -22.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                       Values{11.0, 12.0, 21.0, 22.0});
    target.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3}) -= source;
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 0.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], 0.0);
    EXPECT_EQ(data[4], -11.0);
    EXPECT_EQ(data[5], -12.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], -21.0);
    EXPECT_EQ(data[8], -22.0);
  }
}

TEST(VariableView, nontrivial_slice_minus_equals_slice) {
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source =
        makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                             Values{666.0, 11.0, 12.0, 666.0, 21.0, 22.0});
    target.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2}) -=
        source.slice({Dim::X, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -11.0);
    EXPECT_EQ(data[1], -12.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], -21.0);
    EXPECT_EQ(data[4], -22.0);
    EXPECT_EQ(data[5], 0.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 0.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source =
        makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                             Values{666.0, 11.0, 12.0, 666.0, 21.0, 22.0});
    target.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2}) -=
        source.slice({Dim::X, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], -11.0);
    EXPECT_EQ(data[2], -12.0);
    EXPECT_EQ(data[3], 0.0);
    EXPECT_EQ(data[4], -21.0);
    EXPECT_EQ(data[5], -22.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 0.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source =
        makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                             Values{666.0, 11.0, 12.0, 666.0, 21.0, 22.0});
    target.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3}) -=
        source.slice({Dim::X, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 0.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], -11.0);
    EXPECT_EQ(data[4], -12.0);
    EXPECT_EQ(data[5], 0.0);
    EXPECT_EQ(data[6], -21.0);
    EXPECT_EQ(data[7], -22.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source =
        makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                             Values{666.0, 11.0, 12.0, 666.0, 21.0, 22.0});
    target.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3}) -=
        source.slice({Dim::X, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 0.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], 0.0);
    EXPECT_EQ(data[4], -11.0);
    EXPECT_EQ(data[5], -12.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], -21.0);
    EXPECT_EQ(data[8], -22.0);
  }
}

TEST(VariableView, slice_minus_lower_dimensional) {
  auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
  auto source = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 2.0});
  EXPECT_EQ(target.slice({Dim::Y, 1, 2}).dims(),
            (Dimensions{{Dim::Y, 1}, {Dim::X, 2}}));

  target.slice({Dim::Y, 1, 2}) -= source;

  const auto data = target.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], -2.0);
}

TEST(VariableView, slice_binary_operations) {
  auto v = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                Values{1, 2, 3, 4});
  // Note: There does not seem to be a way to test whether this is using the
  // operators that convert the second argument to Variable (it should not), or
  // keep it as a view. See variable_benchmark.cpp for an attempt to verify
  // this.
  auto sum = v.slice({Dim::X, 0}) + v.slice({Dim::X, 1});
  auto difference = v.slice({Dim::X, 0}) - v.slice({Dim::X, 1});
  auto product = v.slice({Dim::X, 0}) * v.slice({Dim::X, 1});
  auto ratio = v.slice({Dim::X, 0}) / v.slice({Dim::X, 1});
  EXPECT_TRUE(equals(sum.values<double>(), {3, 7}));
  EXPECT_TRUE(equals(difference.values<double>(), {-1, -1}));
  EXPECT_TRUE(equals(product.values<double>(), {2, 12}));
  EXPECT_TRUE(equals(ratio.values<double>(), {1.0 / 2.0, 3.0 / 4.0}));
}

TEST(Variable, reverse) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                  Values{1, 2, 3, 4, 5, 6});
  auto reverseX = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                       Values{3, 2, 1, 6, 5, 4});
  auto reverseY = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                       Values{4, 5, 6, 1, 2, 3});

  EXPECT_EQ(reverse(var, Dim::X), reverseX);
  EXPECT_EQ(reverse(var, Dim::Y), reverseY);
}

TEST(Variable, non_in_place_scalar_operations) {
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2});

  auto sum = var + 1.0;
  EXPECT_TRUE(equals(sum.values<double>(), {2, 3}));
  sum = 2.0 + var;
  EXPECT_TRUE(equals(sum.values<double>(), {3, 4}));

  auto diff = var - 1.0;
  EXPECT_TRUE(equals(diff.values<double>(), {0, 1}));
  diff = 2.0 - var;
  EXPECT_TRUE(equals(diff.values<double>(), {1, 0}));

  auto prod = var * 2.0;
  EXPECT_TRUE(equals(prod.values<double>(), {2, 4}));
  prod = 3.0 * var;
  EXPECT_TRUE(equals(prod.values<double>(), {3, 6}));

  auto ratio = var / 2.0;
  EXPECT_TRUE(equals(ratio.values<double>(), {1.0 / 2.0, 1.0}));
  ratio = 3.0 / var;
  EXPECT_TRUE(equals(ratio.values<double>(), {3.0, 1.5}));
}

TEST(VariableView, scalar_operations) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                  Values{11, 12, 13, 21, 22, 23});

  var.slice({Dim::X, 0}) += 1.0;
  EXPECT_TRUE(equals(var.values<double>(), {12, 12, 13, 22, 22, 23}));
  var.slice({Dim::Y, 1}) += 1.0;
  EXPECT_TRUE(equals(var.values<double>(), {12, 12, 13, 23, 23, 24}));
  var.slice({Dim::X, 1, 3}) += 1.0;
  EXPECT_TRUE(equals(var.values<double>(), {12, 13, 14, 23, 24, 25}));
  var.slice({Dim::X, 1}) -= 1.0;
  EXPECT_TRUE(equals(var.values<double>(), {12, 12, 14, 23, 23, 25}));
  var.slice({Dim::X, 2}) *= 0.0;
  EXPECT_TRUE(equals(var.values<double>(), {12, 12, 0, 23, 23, 0}));
  var.slice({Dim::Y, 0}) /= 2.0;
  EXPECT_TRUE(equals(var.values<double>(), {6, 6, 0, 23, 23, 0}));
}

TEST(VariableTest, binary_op_with_variance) {
  const auto var = makeVariable<double>(
      Dims{Dim::X, Dim::Y}, Shape{2, 3}, Values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
      Variances{0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
  const auto sum = makeVariable<double>(
      Dims{Dim::X, Dim::Y}, Shape{2, 3}, Values{2.0, 4.0, 6.0, 8.0, 10.0, 12.0},
      Variances{0.2, 0.4, 0.6, 0.8, 1.0, 1.2});
  auto tmp = var + var;
  EXPECT_TRUE(tmp.hasVariances());
  EXPECT_EQ(tmp.variances<double>()[0], 0.2);
  EXPECT_EQ(var + var, sum);

  tmp = var * sum;
  EXPECT_EQ(tmp.variances<double>()[0], 0.1 * 2.0 * 2.0 + 0.2 * 1.0 * 1.0);
}

TEST(VariableTest, divide_with_variance) {
  // Note the 0.0: With a wrong implementation the resulting variance is INF.
  const auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0, 0.0},
                                      Variances{0.1, 0.1});
  const auto b = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{3.0, 3.0},
                                      Variances{0.2, 0.2});
  const auto expected =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0 / 3.0, 0.0},
                           Variances{(0.1 / (2.0 * 2.0) + 0.2 / (3.0 * 3.0)) *
                                         (2.0 / 3.0) * (2.0 / 3.0),
                                     /* (0.1 / (0.0 * 0.0) + 0.2 / (3.0
                                      * 3.0)) * (0.0 / 3.0) * (0.0 / 3.0)
                                      naively, but if we take the limit... */
                                     0.1 / (3.0 * 3.0)});
  const auto q = a / b;
  EXPECT_DOUBLE_EQ(q.values<double>()[0], expected.values<double>()[0]);
  EXPECT_DOUBLE_EQ(q.values<double>()[1], expected.values<double>()[1]);
  EXPECT_DOUBLE_EQ(q.variances<double>()[0], expected.variances<double>()[0]);
  EXPECT_DOUBLE_EQ(q.variances<double>()[1], expected.variances<double>()[1]);
}

TEST(VariableTest, boolean_or) {
  const auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, true, true, true});

  const auto result = a | b;

  EXPECT_EQ(result, expected);
}

TEST(VariableTest, boolean_or_equals) {
  auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                              Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  a |= b;
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, true, true, true});

  EXPECT_EQ(a, expected);
}

TEST(VariableTest, boolean_and_equals) {
  auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                              Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  a &= b;
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, false, false, true});

  EXPECT_EQ(a, expected);
}

TEST(VariableTest, boolean_and) {
  const auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, false, false, true});

  const auto result = a & b;

  EXPECT_EQ(result, expected);
}

TEST(VariableTest, boolean_xor_equals) {
  auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                              Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  a ^= b;
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, true, true, false});

  EXPECT_EQ(a, expected);
}

TEST(VariableTest, boolean_xor) {
  const auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, true, true, false});
  const auto result = a ^ b;

  EXPECT_EQ(result, expected);
}

TEST(VariableTest, nan_to_num_throws_when_input_and_replace_types_differ) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, double(NAN)});
  // Replacement type not same as input
  const auto replacement_value = makeVariable<int>(Values{-1});
  EXPECT_THROW(auto replaced = nan_to_num(a, replacement_value),
               except::TypeError);
}

TEST(VariableTest, nan_to_num) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, double(NAN)});
  auto replacement_value = makeVariable<double>(Values{-1});
  Variable b = nan_to_num(a, replacement_value);
  auto expected = makeVariable<double>(
      Dims{Dim::X}, Shape{2}, Values{1.0, replacement_value.value<double>()});
  EXPECT_EQ(b, expected);
}

TEST(VariableTest,
     nan_to_num_with_variance_throws_if_replacement_has_no_variance) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                Values{1.0, double(NAN)}, Variances{0.1, 0.2});

  const auto replacement_value = makeVariable<double>(Values{-1});
  EXPECT_THROW(auto replaced = nan_to_num(a, replacement_value),
               except::VariancesError);
}

TEST(VariableTest, nan_to_num_with_variance_and_variance_on_replacement) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                Values{1.0, double(NAN)}, Variances{0.1, 0.2});

  const auto replacement = makeVariable<double>(Values{-1}, Variances{0.1});
  Variable b = nan_to_num(a, replacement);
  auto expected = makeVariable<double>(
      Dims{Dim::X}, Shape{2}, Values{1.0, replacement.value<double>()},
      Variances{0.1, replacement.variance<double>()});
  EXPECT_EQ(b, expected);
}

TEST(VariableTest,
     nan_to_num_inplace_throws_when_input_and_replace_types_differ) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, double(NAN)});
  // Replacement type not same as input
  const auto replacement_value = makeVariable<int>(Values{-1});
  EXPECT_THROW(nan_to_num(a, replacement_value, a), except::TypeError);
}

TEST(VariableTest,
     nan_to_num_inplace_throws_when_input_and_output_types_differ) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, double(NAN)});
  // Output type not same as input.
  auto out = makeVariable<float>(Dims{Dim::X}, Shape{2}, Values{1.0, 1.0});
  const auto replacement_value = makeVariable<double>(Values{-1});
  EXPECT_THROW(nan_to_num(a, replacement_value, out), except::TypeError);
}

TEST(VariableTest, nan_to_num_inplace) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, double(NAN)});
  const auto replacement_value = makeVariable<double>(Values{-1});
  VariableView b = nan_to_num(a, replacement_value, a);
  auto expected = makeVariable<double>(
      Dims{Dim::X}, Shape{2}, Values{1.0, replacement_value.value<double>()});
  EXPECT_EQ(b, expected);
  EXPECT_EQ(a, expected);
}

TEST(VariableTest,
     nan_to_num_inplace_with_variance_throws_if_replacement_has_no_variance) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{3},
                                Values{1.0, double(NAN), 3.0},
                                Variances{0.1, 0.2, 0.3});
  const auto replacement_value = makeVariable<double>(Values{-1});
  EXPECT_THROW(nan_to_num(a, replacement_value, a), except::VariancesError);
}

TEST(VariableTest, nan_to_num_inplace_out_has_no_variances) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, double(NAN)});
  const auto replacement_value = makeVariable<double>(Values{-1});

  auto out = makeVariable<double>(
      Dims{Dim::X}, Shape{2}, Values{1.0, double(NAN)}, Variances{0.1, 0.2});

  EXPECT_THROW(nan_to_num(a, replacement_value, out), except::VariancesError);
}

TEST(VariableTest,
     nan_to_num_inplace_with_variance_and_variance_on_replacement) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                Values{1.0, double(NAN)}, Variances{0.1, 0.2});
  const auto replacement_value =
      makeVariable<double>(Values{-1}, Variances{0.1});
  VariableView b = nan_to_num(a, replacement_value, a);
  auto expected = makeVariable<double>(
      Dims{Dim::X}, Shape{2}, Values{1.0, replacement_value.value<double>()},
      Variances{0.1, replacement_value.variance<double>()});
  EXPECT_EQ(b, expected);
  EXPECT_EQ(a, expected);
}

template <class T> class ReciprocalTest : public ::testing::Test {};

using test_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ReciprocalTest, test_types);

TYPED_TEST(ReciprocalTest, variable_reciprocal) {
  using T = TypeParam;
  auto var1 = makeVariable<T>(Values{2});
  auto var2 = makeVariable<T>(Values{0.5});
  ASSERT_EQ(reciprocal(var1), var2);
  var1 = makeVariable<T>(Values{2}, Variances{1});
  var2 = makeVariable<T>(Values{0.5}, Variances{0.0625});
  ASSERT_EQ(reciprocal(var1), var2);
}
