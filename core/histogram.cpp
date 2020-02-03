// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/core/histogram.h"
#include "scipp/common/numeric.h"
#include "scipp/core/dataset.h"
#include "scipp/core/except.h"
#include "scipp/core/transform_subspan.h"

#include "dataset_operations_common.h"
#include <iostream>

namespace scipp::core {

static constexpr auto make_histogram = [](auto &data, const auto &events,
                                          const auto &edges) {
  if (scipp::numeric::is_linspace(edges)) {
    // std::cout << "LINSPACE = true" << std::endl;
    // Special implementation for linear bins. Gives a 1x to 20x speedup
    // for few and many events per histogram, respectively.
    const auto [offset, nbin, scale] = linear_edge_params(edges);
    for (const auto &e : events) {
      const double bin = (e - offset) * scale;
      if (bin >= 0.0 && bin < nbin)
        ++data.value[static_cast<scipp::index>(bin)];
    }
  } else {
    // const std::vector<double> new_edges;
    // new_edges.reserve(5000);
    const int new_size = 5000;
    std::vector<scipp::index> bin_number;
    bin_number.reserve(new_size);
    std::vector<bool> contains_edge;
    contains_edge.reserve(new_size);
    const double front = edges.front();
    const auto dx = static_cast<double>(new_size) / (edges.back() - edges.front());
    std::cout << "dx is " << dx << "," << edges.back() << "," << edges.front() << std::endl;
    for (scipp::index i=0; i < new_size; i++)
      contains_edge[i] = false;
    
    // std::generate(new_edges.begin(), new_edges.end(),
    //             [n = edges.front()]() mutable { n += dx ; return n; });
    // auto it = new_edges.begin();
    scipp::index current = 0;
    for (scipp::index i=1; i < edges.size(); i++) {
      const scipp::index ix = static_cast<scipp::index>((edges[i] - front) * dx);
      for (scipp::index j=current; j < ix; j++)
        bin_number[j] = i - 1;
      current = ix;
      contains_edge[ix] = true;
    }

    std::cout << "bin number " << std::endl;
    for (scipp::index i=0; i < 10; i++)
      std::cout << i << "," << bin_number[i] << "," << contains_edge[i] << std::endl;
    std::cout << "======= " << std::endl;
    for (scipp::index i=new_size-10; i < new_size; i++)
      std::cout << i << "," << bin_number[i] << "," << contains_edge[i] << std::endl;


    // const auto [offset, nbin, scale] = linear_edge_params(edges);
    const scipp::index nbin = edges.size();
    for (const auto &e : events) {
      const scipp::index ib = static_cast<scipp::index>((e - front) * dx);
      scipp::index bin = bin_number[ib];
      // if (contains_edge[ib] && e < edges[bin])
      //   bin--;

      if (bin >= 0 && bin < nbin){
        if (contains_edge[ib])
          std::cout << e << "," << ib << "," << bin << " This high res bin contains an edge " << edges[bin] << std::endl;
          
        if (contains_edge[ib] && e < edges[bin]) {
          std::cout << "event " << e << "has bin number " << bin << " and bin index " << ib;
          std::cout << ". The event is below the edge, so we decrease bin" << std::endl;
          --bin;
        }
        ++data.value[bin];
      }
    }



  }




  // else if (scipp::numeric::is_logspace(edges)) {
  //   // std::cout << "LOGSPACE = true" << std::endl;
  //   // Special implementation for log bins.
  //   const auto [offset, nbin, scale] = log_edge_params(edges);
  //   for (const auto &e : events) {
  //     const double bin = (e - offset) * scale;
  //     // const double bin = (std::log(e) - offset) * scale;
  //     if (bin >= 0.0 && bin < nbin)
  //       ++data.value[static_cast<scipp::index>(bin)];
  //   }
  // } else {
  //   // std::cout << "DEFAULT" << std::endl;
  //   expect::histogram::sorted_edges(edges);
  //   for (const auto &e : events) {
  //     auto it = std::upper_bound(edges.begin(), edges.end(), e);
  //     if (it != edges.end() && it != edges.begin())
  //       ++data.value[--it - edges.begin()];
  //   }
  // }
  std::copy(data.value.begin(), data.value.end(), data.variance.begin());
};

static constexpr auto make_histogram_from_weighted =
    [](auto &data, const auto &events, const auto &weights, const auto &edges) {
      if (scipp::numeric::is_linspace(edges)) {
        const auto [offset, nbin, scale] = linear_edge_params(edges);
        for (scipp::index i = 0; i < scipp::size(events); ++i) {
          const auto x = events[i];
          const double bin = (x - offset) * scale;
          if (bin >= 0.0 && bin < nbin) {
            const auto b = static_cast<scipp::index>(bin);
            const auto w = weights.values[i];
            const auto e = weights.variances[i];
            data.value[b] += w;
            data.variance[b] += e;
          }
        }
      } else if (scipp::numeric::is_logspace(edges)) {
        const auto [offset, nbin, scale] = log_edge_params(edges);
        for (scipp::index i = 0; i < scipp::size(events); ++i) {
          // const auto x = std::log(events[i]);
          const auto x = events[i];
          const double bin = (x - offset) * scale;
          if (bin >= 0.0 && bin < nbin) {
            const auto b = static_cast<scipp::index>(bin);
            const auto w = weights.values[i];
            const auto e = weights.variances[i];
            data.value[b] += w;
            data.variance[b] += e;
          }
        }
      } else {
        expect::histogram::sorted_edges(edges);
        for (scipp::index i = 0; i < scipp::size(events); ++i) {
          const auto x = events[i];
          auto it = std::upper_bound(edges.begin(), edges.end(), x);
          if (it != edges.end() && it != edges.begin()) {
            const auto b = --it - edges.begin();
            const auto w = weights.values[i];
            const auto e = weights.variances[i];
            data.value[b] += w;
            data.variance[b] += e;
          }
        }
      }
    };

static constexpr auto make_histogram_unit = [](const units::Unit &sparse_unit,
                                               const units::Unit &edge_unit) {
  if (sparse_unit != edge_unit)
    throw except::UnitError("Bin edges must have same unit as the sparse "
                            "input coordinate.");
  return units::counts;
};

static constexpr auto make_histogram_unit_from_weighted =
    [](const units::Unit &sparse_unit, const units::Unit &weights_unit,
       const units::Unit &edge_unit) {
      if (sparse_unit != edge_unit)
        throw except::UnitError("Bin edges must have same unit as the sparse "
                                "input coordinate.");
      if (weights_unit != units::counts && weights_unit != units::dimensionless)
        throw except::UnitError("Weights of sparse data must be "
                                "`units::counts` or `units::dimensionless`.");
      return weights_unit;
    };

namespace histogram_detail {
template <class Out, class Coord, class Edge>
using args = std::tuple<span<Out>, sparse_container<Coord>, span<const Edge>>;
}
namespace histogram_weighted_detail {
template <class Out, class Coord, class Weight, class Edge>
using args = std::tuple<span<Out>, sparse_container<Coord>,
                        sparse_container<Weight>, span<const Edge>>;
}

DataArray histogram(const DataConstProxy &sparse,
                    const VariableConstProxy &binEdges) {
  auto dim = binEdges.dims().inner();

  auto result = apply_and_drop_dim(
      sparse,
      [](const DataConstProxy &sparse_, const Dim dim_,
         const VariableConstProxy &binEdges_) {
        if (sparse_.hasData()) {
          using namespace histogram_weighted_detail;
          return transform_subspan<
              std::tuple<args<double, double, double, double>,
                         args<double, float, double, double>,
                         args<double, float, double, float>,
                         args<double, double, float, double>>>(
              dim_, binEdges_.dims()[dim_] - 1, sparse_.coords()[dim_],
              sparse_.data(), binEdges_,
              overloaded{make_histogram_from_weighted,
                         make_histogram_unit_from_weighted,
                         transform_flags::expect_variance_arg<0>,
                         transform_flags::expect_no_variance_arg<1>,
                         transform_flags::expect_variance_arg<2>,
                         transform_flags::expect_no_variance_arg<3>});
        } else {
          using namespace histogram_detail;
          return transform_subspan<std::tuple<args<double, double, double>,
                                              args<double, float, double>,
                                              args<double, float, float>>>(
              dim_, binEdges_.dims()[dim_] - 1, sparse_.coords()[dim_],
              binEdges_,
              overloaded{make_histogram, make_histogram_unit,
                         transform_flags::expect_variance_arg<0>,
                         transform_flags::expect_no_variance_arg<1>,
                         transform_flags::expect_no_variance_arg<2>});
        }
      },
      dim, binEdges);
  result.setCoord(dim, binEdges);
  return result;
}

DataArray histogram(const DataConstProxy &sparse, const Variable &binEdges) {
  return histogram(sparse, VariableConstProxy(binEdges));
}

Dataset histogram(const Dataset &dataset, const VariableConstProxy &bins) {
  auto out(Dataset(DatasetConstProxy::makeProxyWithEmptyIndexes(dataset)));
  out.setCoord(bins.dims().inner(), bins);
  for (const auto &item : dataset) {
    if (item.dims().sparse())
      out.setData(item.name(), histogram(item, bins));
  }
  return out;
}

Dataset histogram(const Dataset &dataset, const Variable &bins) {
  return histogram(dataset, VariableConstProxy(bins));
}

Dataset histogram(const Dataset &dataset, const Dim &dim) {
  auto bins = dataset.coords()[dim];
  return histogram(dataset, bins);
}

/// Return true if the data array respresents a histogram for given dim.
bool is_histogram(const DataConstProxy &a, const Dim dim) {
  const auto dims = a.dims();
  const auto coords = a.coords();
  return !dims.sparse() && dims.contains(dim) && coords.contains(dim) &&
         coords[dim].dims().contains(dim) &&
         coords[dim].dims()[dim] == dims[dim] + 1;
}

} // namespace scipp::core
