// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/dataset/groupby.h"
#include "scipp/dataset/unaligned.h"
#include "scipp/variable/binary_arithmetic.h"

#include "test_macros.h"

using namespace scipp;
using namespace scipp::dataset;

struct GroupbyTest : public ::testing::Test {
  GroupbyTest() {
    d.setData("a",
              makeVariable<double>(Dimensions{Dim::X, 3}, units::Unit(units::m),
                                   Values{1, 2, 3}, Variances{4, 5, 6}));
    d.setData("b",
              makeVariable<double>(Dimensions{Dim::X, 3}, units::Unit(units::s),
                                   Values{0.1, 0.2, 0.3}));
    d.setData("c", makeVariable<double>(Dimensions{{Dim::Z, 2}, {Dim::X, 3}},
                                        units::Unit(units::s),
                                        Values{1, 2, 3, 4, 5, 6}));
    d.setAttr("a", "scalar", makeVariable<double>(Values{1.2}));
    d.setCoord(Dim("labels1"),
               makeVariable<double>(Dimensions{Dim::X, 3},
                                    units::Unit(units::m), Values{1, 2, 3}));
    d.setCoord(Dim("labels2"),
               makeVariable<double>(Dimensions{Dim::X, 3},
                                    units::Unit(units::m), Values{1, 1, 3}));
  }

  Dataset d;
};

TEST_F(GroupbyTest, fail_key_not_found) {
  EXPECT_THROW(groupby(d, Dim("invalid")), except::NotFoundError);
  EXPECT_THROW(groupby(d["a"], Dim("invalid")), except::NotFoundError);
}

TEST_F(GroupbyTest, fail_key_2d) {
  d.setCoord(Dim("2d"), makeVariable<double>(
                            Dimensions{{Dim::Z, 2}, {Dim::X, 3}},
                            units::Unit(units::s), Values{1, 2, 3, 4, 5, 6}));
  EXPECT_THROW(groupby(d, Dim("2d")), except::DimensionError);
  EXPECT_THROW(groupby(d["a"], Dim("2d")), except::DimensionError);
}

TEST_F(GroupbyTest, fail_key_with_variances) {
  d.setCoord(Dim("variances"),
             makeVariable<double>(Dimensions{Dim::X, 3}, units::Unit(units::m),
                                  Values{1, 2, 3}, Variances{4, 5, 6}));
  EXPECT_THROW(groupby(d, Dim("variances")), except::VariancesError);
  EXPECT_THROW(groupby(d["a"], Dim("variances")), except::VariancesError);
}

TEST_F(GroupbyTest, copy) {
  auto one_group =
      groupby(d, Dim("labels1"),
              makeVariable<double>(Dims{Dim("labels1")}, Shape{2},
                                   units::Unit(units::m), Values{0, 4}));
  EXPECT_EQ(one_group.size(), 1);
  EXPECT_EQ(one_group.copy(0), d);

  auto two_groups =
      groupby(d, Dim("labels1"),
              makeVariable<double>(Dims{Dim("labels1")}, Shape{3},
                                   units::Unit(units::m), Values{0, 3, 4}));
  EXPECT_EQ(two_groups.size(), 2);
  EXPECT_EQ(two_groups.copy(0), d.slice({Dim::X, 0, 2}));
  EXPECT_EQ(two_groups.copy(1), d.slice({Dim::X, 2, 3}));
}

TEST_F(GroupbyTest, dataset_1d_and_2d) {
  Dataset expected;
  Dim dim("labels2");
  expected.setData(
      "a", makeVariable<double>(Dims{dim}, Shape{2}, units::Unit(units::m),
                                Values{1.5, 3.0}, Variances{9.0 / 4, 6.0}));
  expected.setData("b", makeVariable<double>(Dims{dim}, Shape{2},
                                             units::Unit(units::s),
                                             Values{(0.1 + 0.2) / 2.0, 0.3}));
  expected.setData("c", makeVariable<double>(Dims{Dim(Dim::Z), dim},
                                             Shape{2, 2}, units::Unit(units::s),
                                             Values{1.5, 3.0, 4.5, 6.0}));
  expected.setAttr("a", "scalar", makeVariable<double>(Values{1.2}));
  expected.setCoord(dim,
                    makeVariable<double>(Dims{dim}, Shape{2},
                                         units::Unit(units::m), Values{1, 3}));

  EXPECT_EQ(groupby(d, dim).mean(Dim::X), expected);
  EXPECT_EQ(groupby(d["a"], dim).mean(Dim::X), expected["a"]);
  EXPECT_EQ(groupby(d["b"], dim).mean(Dim::X), expected["b"]);
  EXPECT_EQ(groupby(d["c"], dim).mean(Dim::X), expected["c"]);
}

struct GroupbyMaskedTest : public GroupbyTest {
  GroupbyMaskedTest() : GroupbyTest() {
    d.setMask("mask_x", makeVariable<bool>(Dimensions{Dim::X, 3},
                                           Values{false, true, false}));
    d.setMask("mask_z",
              makeVariable<bool>(Dimensions{Dim::Z, 2}, Values{false, true}));
  }
};

TEST_F(GroupbyMaskedTest, sum) {
  Dataset expected;
  const Dim dim("labels2");
  expected.setData("a", makeVariable<double>(Dimensions{dim, 2},
                                             units::Unit(units::m),
                                             Values{1, 3}, Variances{4, 6}));
  expected.setData("b", makeVariable<double>(Dimensions{dim, 2},
                                             units::Unit(units::s),
                                             Values{0.1, 0.3}));
  expected.setData("c", makeVariable<double>(Dimensions{{Dim::Z, 2}, {dim, 2}},
                                             units::Unit(units::s),
                                             Values{1, 3, 4, 6}));
  expected.setCoord(dim,
                    makeVariable<double>(Dimensions{dim, 2},
                                         units::Unit(units::m), Values{1, 3}));
  expected.setAttr("a", "scalar", makeVariable<double>(Values{1.2}));
  expected.setMask(
      "mask_z", makeVariable<bool>(Dimensions{Dim::Z, 2}, Values{false, true}));

  const auto result = groupby(d, dim).sum(Dim::X);
  EXPECT_EQ(result, expected);
}

TEST_F(GroupbyMaskedTest, sum_irrelvant_mask) {
  Dataset expected;
  const Dim dim("labels2");
  expected.setData("a", makeVariable<double>(Dimensions{dim, 2},
                                             units::Unit(units::m),
                                             Values{3, 3}, Variances{9, 6}));
  expected.setData("b", makeVariable<double>(Dimensions{dim, 2},
                                             units::Unit(units::s),
                                             Values{0.1 + 0.2, 0.3}));
  expected.setData("c", makeVariable<double>(Dimensions{{Dim::Z, 2}, {dim, 2}},
                                             units::Unit(units::s),
                                             Values{3, 3, 9, 6}));
  expected.setCoord(dim,
                    makeVariable<double>(Dimensions{dim, 2},
                                         units::Unit(units::m), Values{1, 3}));
  expected.setAttr("a", "scalar", makeVariable<double>(Values{1.2}));
  expected.setMask(
      "mask_z", makeVariable<bool>(Dimensions{Dim::Z, 2}, Values{false, true}));

  d.masks().erase("mask_x");
  auto result = groupby(d, dim).sum(Dim::X);
  EXPECT_EQ(result, expected);

  d.masks().erase("mask_z");
  ASSERT_TRUE(d.masks().empty());
  const auto expected2 = groupby(d, dim).sum(Dim::X);
  result.masks().erase("mask_z");
  EXPECT_EQ(result, expected2);
}

TEST_F(GroupbyMaskedTest, mean_mask_ignores_values_properly) {
  // the mask is on a coordinate that the label does not include
  // this test verifies that the data is not affected
  Dataset expected;
  const Dim dim("labels2");
  expected.setData("a", makeVariable<double>(Dimensions{dim, 2},
                                             units::Unit(units::m),
                                             Values{1, 3}, Variances{4, 6}));
  expected.setData("b", makeVariable<double>(Dimensions{dim, 2},
                                             units::Unit(units::s),
                                             Values{0.1, 0.3}));
  expected.setData("c", makeVariable<double>(Dimensions{{Dim::Z, 2}, {dim, 2}},
                                             units::Unit(units::s),
                                             Values{1, 3, 4, 6}));
  expected.setCoord(dim,
                    makeVariable<double>(Dimensions{dim, 2},
                                         units::Unit(units::m), Values{1, 3}));
  expected.setAttr("a", "scalar", makeVariable<double>(Values{1.2}));
  expected.setMask(
      "mask_z", makeVariable<bool>(Dimensions{Dim::Z, 2}, Values{false, true}));

  const auto result = groupby(d, dim).mean(Dim::X);
  EXPECT_EQ(result, expected);
}

TEST_F(GroupbyMaskedTest, mean) {
  const auto result = groupby(d, Dim("labels1")).mean(Dim::X);

  EXPECT_EQ(result["a"].template values<double>()[0], 1.0);
  EXPECT_TRUE(std::isnan(result["a"].template values<double>()[1]));
  EXPECT_EQ(result["a"].template values<double>()[2], 3.0);

  EXPECT_EQ(result["a"].template variances<double>()[0], 4.0);
  EXPECT_TRUE(std::isnan(result["a"].template variances<double>()[1]));
  EXPECT_EQ(result["a"].template variances<double>()[2], 6.0);

  EXPECT_EQ(result["b"].template values<double>()[0], 0.1);
  EXPECT_TRUE(std::isnan(result["b"].template values<double>()[1]));
  EXPECT_EQ(result["b"].template values<double>()[2], 0.3);

  EXPECT_EQ(result["c"].template values<double>()[0], 1.0);
  EXPECT_TRUE(std::isnan(result["c"].template values<double>()[1]));
  EXPECT_EQ(result["c"].template values<double>()[2], 3.0);
  EXPECT_EQ(result["c"].template values<double>()[3], 4.0);
  EXPECT_TRUE(std::isnan(result["c"].template values<double>()[4]));
  EXPECT_EQ(result["c"].template values<double>()[5], 6.0);
}

TEST_F(GroupbyMaskedTest, mean2) {
  d.setMask("mask_x", makeVariable<bool>(Dimensions{Dim::X, 3},
                                         Values{false, false, true}));

  const Dim dim("labels2");
  const auto result = groupby(d, dim).mean(Dim::X);

  EXPECT_EQ(result["a"].template values<double>()[0], 1.5);
  EXPECT_TRUE(std::isnan(result["a"].template values<double>()[1]));
  EXPECT_EQ(result["a"].template variances<double>()[0], 2.25);
  EXPECT_TRUE(std::isnan(result["a"].template variances<double>()[1]));

  EXPECT_DOUBLE_EQ(result["b"].template values<double>()[0], 0.15);
  EXPECT_TRUE(std::isnan(result["b"].template values<double>()[1]));

  EXPECT_EQ(result["c"].template values<double>()[0], 1.5);
  EXPECT_TRUE(std::isnan(result["c"].template values<double>()[1]));
  EXPECT_EQ(result["c"].template values<double>()[2], 4.5);
  EXPECT_TRUE(std::isnan(result["c"].template values<double>()[3]));

  EXPECT_EQ(result.coords()[dim],
            makeVariable<double>(Dimensions{dim, 2}, units::Unit(units::m),
                                 Values{1.0, 3.0}));
}

TEST(GroupbyMaskedDataArrayTest, sum) {
  DataArray arr{
      makeVariable<int>(Dimensions{{Dim::Y, 2}, {Dim::X, 3}},
                        Values{1, 2, 3, 4, 5, 6}),
      {{Dim::Y, makeVariable<int>(Dimensions{Dim::Y, 2}, Values{1, 2})},
       {Dim::X, makeVariable<int>(Dimensions{Dim::X, 3}, Values{1, 2, 3})},
       {Dim("labels"),
        makeVariable<double>(Dimensions{Dim::X, 3}, Values{1, 1, 3})}},
      {{"masks", makeVariable<bool>(Dimensions{Dim::X, 3},
                                    Values{false, true, false})}}};

  const Dim dim("labels");
  DataArray expected{
      makeVariable<int>(Dimensions{{Dim::Y, 2}, {dim, 2}}, Values{1, 3, 4, 6}),
      {{Dim::Y, makeVariable<int>(Dimensions{Dim::Y, 2}, Values{1, 2})},
       {dim, makeVariable<double>(Dimensions{dim, 2}, Values{1, 3})}}};

  EXPECT_EQ(groupby(arr, dim).sum(Dim::X), expected);
}

TEST(GroupbyMaskedDataArrayTest, mean) {
  DataArray arr{
      makeVariable<int>(Dimensions{{Dim::Y, 2}, {Dim::X, 3}},
                        Values{1, 2, 3, 4, 5, 6}),
      {{Dim::Y, makeVariable<int>(Dimensions{Dim::Y, 2}, Values{1, 2})},
       {Dim::X, makeVariable<int>(Dimensions{Dim::X, 3}, Values{1, 2, 3})},
       {Dim("labels"),
        makeVariable<double>(Dimensions{Dim::X, 3}, Values{1, 2, 3})}},
      {{"masks", makeVariable<bool>(Dimensions{Dim::X, 3},
                                    Values{false, true, false})}}};

  const auto result = groupby(arr, Dim("labels")).mean(Dim::X);

  EXPECT_EQ(result.template values<double>()[0], 1.0);
  EXPECT_TRUE(std::isnan(result.template values<double>()[1]));
  EXPECT_EQ(result.template values<double>()[2], 3.0);
  EXPECT_EQ(result.template values<double>()[3], 4.0);
  EXPECT_TRUE(std::isnan(result.template values<double>()[4]));
  EXPECT_EQ(result.template values<double>()[5], 6.0);
}

TEST(GroupbyMaskedDataArrayTest, mean2) {
  DataArray arr{
      makeVariable<int>(Dimensions{{Dim::Y, 2}, {Dim::X, 3}},
                        Values{1, 2, 3, 4, 5, 6}),
      {{Dim::Y, makeVariable<int>(Dimensions{Dim::Y, 2}, Values{1, 2})},
       {Dim::X, makeVariable<int>(Dimensions{Dim::X, 3}, Values{1, 2, 3})},
       {Dim("labels"),
        makeVariable<double>(Dimensions{Dim::X, 3}, Values{1, 1, 3})}},
      {{"masks", makeVariable<bool>(Dimensions{Dim::X, 3},
                                    Values{false, false, true})}}};

  const auto result = groupby(arr, Dim("labels")).mean(Dim::X);

  EXPECT_EQ(result.template values<double>()[0], 1.5);
  EXPECT_TRUE(std::isnan(result.template values<double>()[1]));
  EXPECT_EQ(result.template values<double>()[2], 4.5);
  EXPECT_TRUE(std::isnan(result.template values<double>()[3]));
}

struct GroupbyWithBinsTest : public ::testing::Test {
  GroupbyWithBinsTest() {
    d.setData("a",
              makeVariable<double>(Dimensions{Dim::X, 5}, units::Unit(units::s),
                                   Values{0.1, 0.2, 0.3, 0.4, 0.5}));
    d.setData("b", makeVariable<double>(Dimensions{{Dim::Y, 2}, {Dim::X, 5}},
                                        units::Unit(units::s),
                                        Values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    d.setAttr("a", "scalar", makeVariable<double>(Values{1.2}));
    d.setCoord(Dim("labels1"), makeVariable<double>(Dimensions{Dim::X, 5},
                                                    units::Unit(units::m),
                                                    Values{1, 2, 3, 4, 5}));
    d.setCoord(Dim("labels2"), makeVariable<double>(
                                   Dimensions{Dim::X, 5}, units::Unit(units::m),
                                   Values{1.0, 1.1, 2.5, 4.0, 1.2}));
  }

  Dataset d;
};

TEST_F(GroupbyWithBinsTest, bins) {
  auto bins =
      makeVariable<double>(Dims{Dim::Z}, Shape{4}, units::Unit(units::m),
                           Values{0.0, 1.0, 2.0, 3.0});

  Dataset expected;
  expected.setCoord(Dim::Z, bins);
  expected.setData("a", makeVariable<double>(Dims{Dim::Z}, Shape{3},
                                             units::Unit(units::s),
                                             Values{0.0, 0.8, 0.3}));
  expected.setData("b", makeVariable<double>(Dims{Dim::Y, Dim::Z}, Shape{2, 3},
                                             units::Unit(units::s),
                                             Values{0, 8, 3, 0, 23, 8}));
  expected.setAttr("a", "scalar", makeVariable<double>(Values{1.2}));

  EXPECT_EQ(groupby(d, Dim("labels2"), bins).sum(Dim::X), expected);
  EXPECT_EQ(groupby(d["a"], Dim("labels2"), bins).sum(Dim::X), expected["a"]);
  EXPECT_EQ(groupby(d["b"], Dim("labels2"), bins).sum(Dim::X), expected["b"]);
}

TEST_F(GroupbyWithBinsTest, bins_mean_empty) {
  auto bins =
      makeVariable<double>(Dims{Dim::Z}, Shape{4}, units::Unit(units::m),
                           Values{0.0, 1.0, 2.0, 3.0});

  const auto binned = groupby(d, Dim("labels2"), bins).mean(Dim::X);
  EXPECT_TRUE(std::isnan(binned["a"].values<double>()[0]));
  EXPECT_FALSE(std::isnan(binned["a"].values<double>()[1]));
  EXPECT_TRUE(std::isnan(binned["b"].values<double>()[0]));
  EXPECT_TRUE(std::isnan(binned["b"].values<double>()[3]));
  EXPECT_FALSE(std::isnan(binned["b"].values<double>()[1]));
}

TEST_F(GroupbyWithBinsTest, single_bin) {
  auto bins = makeVariable<double>(Dims{Dim::Z}, Shape{2},
                                   units::Unit(units::m), Values{1.0, 5.0});
  const auto groups = groupby(d, Dim("labels2"), bins);

  // Non-range slice drops Dim::Z and the corresponding coord (the edges), so
  // the result must be equal to a global `sum` or `mean`.
  EXPECT_EQ(groups.sum(Dim::X).slice({Dim::Z, 0}), sum(d, Dim::X));
  EXPECT_EQ(groups.mean(Dim::X).slice({Dim::Z, 0}), mean(d, Dim::X));
}

TEST_F(GroupbyWithBinsTest, two_bin) {
  auto bins = makeVariable<double>(
      Dims{Dim::Z}, Shape{3}, units::Unit(units::m), Values{1.0, 2.0, 5.0});
  const auto groups = groupby(d, Dim("labels2"), bins);

  auto group0 =
      concatenate(d.slice({Dim::X, 0, 2}), d.slice({Dim::X, 4, 5}), Dim::X);
  // concatenate does currently not preserve attributes
  group0.setAttr("a", "scalar", d["a"].attrs()["scalar"]);
  EXPECT_EQ(groups.sum(Dim::X).slice({Dim::Z, 0}), sum(group0, Dim::X));
  EXPECT_EQ(groups.mean(Dim::X).slice({Dim::Z, 0}), mean(group0, Dim::X));

  const auto group1 = d.slice({Dim::X, 2, 4});
  EXPECT_EQ(groups.sum(Dim::X).slice({Dim::Z, 1}), sum(group1, Dim::X));
  EXPECT_EQ(groups.mean(Dim::X).slice({Dim::Z, 1}), mean(group1, Dim::X));
}

auto make_events_in() {
  auto var = makeVariable<event_list<double>>(Dims{Dim::Y}, Shape{3});
  const auto &var_ = var.values<event_list<double>>();
  var_[0] = {1, 2, 3};
  var_[1] = {4, 5};
  var_[2] = {6, 7};
  return var;
}

auto make_events_out(bool mask = false) {
  auto var = makeVariable<event_list<double>>(Dims{Dim("labels")}, Shape{2});
  const auto &var_ = var.values<event_list<double>>();
  if (mask)
    var_[0] = {1, 2, 3};
  else
    var_[0] = {1, 2, 3, 4, 5};
  var_[1] = {6, 7};
  return var;
}

struct GroupbyFlattenDefaultWeight : public ::testing::Test {
  const DataArray a{
      makeVariable<double>(Dims{Dim::Y}, Shape{3}, units::Unit(units::counts),
                           Values{1, 1, 1}, Variances{1, 1, 1}),
      {{Dim::X, make_events_in()},
       {Dim("0-d"), makeVariable<double>(Values{1.2})},
       {Dim("labels"),
        makeVariable<double>(Dims{Dim::Y}, Shape{3}, units::Unit(units::m),
                             Values{1, 1, 3})},
       {Dim("dense"),
        makeVariable<double>(Dims{Dim::X}, Shape{5}, units::Unit(units::m),
                             Values{1, 2, 3, 4, 5})}},
      {},
      {{"scalar_attr", makeVariable<double>(Values{1.2})}}};

  const DataArray expected{
      makeVariable<double>(Dims{Dim("labels")}, Shape{2},
                           units::Unit(units::counts), Values{1, 1},
                           Variances{1, 1}),
      {{Dim::X, make_events_out()},
       {Dim("0-d"), makeVariable<double>(Values{1.2})},
       {Dim("labels"),
        makeVariable<double>(Dims{Dim("labels")}, Shape{2},
                             units::Unit(units::m), Values{1, 3})},
       {Dim("dense"),
        makeVariable<double>(Dims{Dim::X}, Shape{5}, units::Unit(units::m),
                             Values{1, 2, 3, 4, 5})}},
      {},
      {{"scalar_attr", makeVariable<double>(Values{1.2})}}};
};

TEST_F(GroupbyFlattenDefaultWeight, flatten_coord_only) {
  EXPECT_EQ(groupby(a, Dim("labels")).flatten(Dim::Y), expected);
}

TEST_F(GroupbyFlattenDefaultWeight, sum_realigned_coord_only) {
  const auto edges =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{0, 10});
  const auto realigned = unaligned::realign(a, {{Dim::X, edges}});

  const auto summed = groupby(realigned, Dim("labels")).sum(Dim::Y);
  EXPECT_EQ(summed.unaligned(), expected);
}

TEST_F(GroupbyFlattenDefaultWeight, flatten_dataset_coord_only) {
  const Dataset d{{{"a", a}, {"b", a}}};
  const Dataset expected_d{{{"a", expected}, {"b", expected}}};
  EXPECT_EQ(groupby(d, Dim("labels")).flatten(Dim::Y), expected_d);
}

TEST_F(GroupbyFlattenDefaultWeight, flatten_non_constant_scalar_weight_fail) {
  Dataset d{{{"a", a}, {"b", a}}};
  d["a"].values<double>()[0] += 0.1;
  EXPECT_THROW(groupby(d, Dim("labels")).flatten(Dim::Y),
               except::EventDataError);
}

TEST(GroupbyFlattenTest, flatten_coord_and_labels) {
  DataArray a{makeVariable<double>(Dims{Dim::Y}, Shape{3},
                                   units::Unit(units::counts), Values{1, 1, 1},
                                   Variances{1, 1, 1}),
              {{Dim::X, make_events_in()},
               {Dim("events"),
                make_events_in() * (0.3 * units::Unit(units::dimensionless))},
               {Dim("labels"),
                makeVariable<double>(Dims{Dim::Y}, Shape{3},
                                     units::Unit(units::m), Values{1, 1, 3})}}};

  DataArray expected{
      makeVariable<double>(Dims{Dim("labels")}, Shape{2},
                           units::Unit(units::counts), Values{1, 1},
                           Variances{1, 1}),
      {{Dim::X, make_events_out()},
       {Dim("labels"),
        makeVariable<double>(Dims{Dim("labels")}, Shape{2},
                             units::Unit(units::m), Values{1, 3})},
       {Dim("events"),
        make_events_out() * (0.3 * units::Unit(units::dimensionless))}}};

  EXPECT_EQ(groupby(a, Dim("labels")).flatten(Dim::Y), expected);
}

TEST(GroupbyFlattenTest, flatten_coord_and_data) {
  DataArray a{make_events_in() * (1.5 * units::Unit(units::dimensionless)),
              {{Dim::X, make_events_in()},
               {Dim::Y, makeVariable<double>(Dims{Dim::Y}, Shape{3})},
               {Dim("labels"),
                makeVariable<double>(Dims{Dim::Y}, Shape{3},
                                     units::Unit(units::m), Values{1, 1, 3})}}};

  DataArray expected{
      make_events_out() * (1.5 * units::Unit(units::dimensionless)),
      {{Dim::X, make_events_out()},
       {Dim("labels"),
        makeVariable<double>(Dims{Dim("labels")}, Shape{2},
                             units::Unit(units::m), Values{1, 3})}}};

  EXPECT_EQ(groupby(a, Dim("labels")).flatten(Dim::Y), expected);
}

TEST(GroupbyFlattenTest, flatten_with_mask) {
  DataArray a{make_events_in() * (1.5 * units::Unit(units::dimensionless)),
              {{Dim::X, make_events_in()},
               {Dim::Y, makeVariable<double>(Dims{Dim::Y}, Shape{3})},
               {Dim("labels"),
                makeVariable<double>(Dims{Dim::Y}, Shape{3},
                                     units::Unit(units::m), Values{1, 1, 3})}},
              {{"mask_y", makeVariable<bool>(Dims{Dim::Y}, Shape{3},
                                             Values{false, true, false})}}};

  bool mask = true;
  DataArray expected{
      make_events_out(mask) * (1.5 * units::Unit(units::dimensionless)),
      {{Dim::X, make_events_out(mask)},
       {Dim("labels"),
        makeVariable<double>(Dims{Dim("labels")}, Shape{2},
                             units::Unit(units::m), Values{1, 3})}}};

  EXPECT_EQ(groupby(a, Dim("labels")).flatten(Dim::Y), expected);
}

struct GroupbyLogicalTest : public ::testing::Test {
  GroupbyLogicalTest() {
    d.setData(
        "a", makeVariable<bool>(Dimensions{{Dim::Z, 2}, {Dim::X, 3}},
                                Values{true, false, false, true, true, false}));
    d.setCoord(Dim("labels1"),
               makeVariable<double>(Dimensions{Dim::X, 3},
                                    units::Unit(units::m), Values{1, 2, 3}));
    d.setCoord(Dim("labels2"),
               makeVariable<double>(Dimensions{Dim::X, 3},
                                    units::Unit(units::m), Values{1, 1, 3}));
  }
  Dataset d;
};

TEST_F(GroupbyLogicalTest, no_reduction) {
  Dataset expected(d);
  expected.rename(Dim::X, Dim("labels1"));
  expected.setCoord(Dim("labels1"), expected.coords()[Dim("labels1")]);
  expected.coords().erase(Dim("labels2"));
  EXPECT_EQ(groupby(d, Dim("labels1")).all(Dim::X), expected);
  EXPECT_EQ(groupby(d, Dim("labels1")).any(Dim::X), expected);
}

TEST_F(GroupbyLogicalTest, all) {
  Dataset expected;
  expected.setData(
      "a", makeVariable<bool>(Dimensions{{Dim::Z, 2}, {Dim("labels2"), 2}},
                              Values{false, false, true, false}));
  expected.setCoord(Dim("labels2"),
                    makeVariable<double>(Dimensions{Dim("labels2"), 2},
                                         units::Unit(units::m), Values{1, 3}));
  EXPECT_EQ(groupby(d, Dim("labels2")).all(Dim::X), expected);
}

TEST_F(GroupbyLogicalTest, any) {
  Dataset expected;
  expected.setData(
      "a", makeVariable<bool>(Dimensions{{Dim::Z, 2}, {Dim("labels2"), 2}},
                              Values{true, false, true, false}));
  expected.setCoord(Dim("labels2"),
                    makeVariable<double>(Dimensions{Dim("labels2"), 2},
                                         units::Unit(units::m), Values{1, 3}));
  EXPECT_EQ(groupby(d, Dim("labels2")).any(Dim::X), expected);
}

struct GroupbyMinMaxTest : public ::testing::Test {
  GroupbyMinMaxTest() {
    d.setData("a", makeVariable<double>(Dimensions{{Dim::Z, 2}, {Dim::X, 3}},
                                        Values{1, 2, 3, 4, 5, 6}));
    d.setCoord(Dim("labels1"),
               makeVariable<double>(Dimensions{Dim::X, 3},
                                    units::Unit(units::m), Values{1, 2, 3}));
    d.setCoord(Dim("labels2"),
               makeVariable<double>(Dimensions{Dim::X, 3},
                                    units::Unit(units::m), Values{1, 1, 3}));
  }
  Dataset d;
};

TEST_F(GroupbyMinMaxTest, no_reduction) {
  Dataset expected(d);
  expected.rename(Dim::X, Dim("labels1"));
  expected.setCoord(Dim("labels1"), expected.coords()[Dim("labels1")]);
  expected.coords().erase(Dim("labels2"));
  EXPECT_EQ(groupby(d, Dim("labels1")).min(Dim::X), expected);
  EXPECT_EQ(groupby(d, Dim("labels1")).max(Dim::X), expected);
}

TEST_F(GroupbyMinMaxTest, min) {
  Dataset expected;
  expected.setData(
      "a", makeVariable<double>(Dimensions{{Dim::Z, 2}, {Dim("labels2"), 2}},
                                Values{1, 3, 4, 6}));
  expected.setCoord(Dim("labels2"),
                    makeVariable<double>(Dimensions{Dim("labels2"), 2},
                                         units::Unit(units::m), Values{1, 3}));
  EXPECT_EQ(groupby(d, Dim("labels2")).min(Dim::X), expected);
}

TEST_F(GroupbyMinMaxTest, max) {
  Dataset expected;
  expected.setData(
      "a", makeVariable<double>(Dimensions{{Dim::Z, 2}, {Dim("labels2"), 2}},
                                Values{2, 3, 5, 6}));
  expected.setCoord(Dim("labels2"),
                    makeVariable<double>(Dimensions{Dim("labels2"), 2},
                                         units::Unit(units::m), Values{1, 3}));
  EXPECT_EQ(groupby(d, Dim("labels2")).max(Dim::X), expected);
}
