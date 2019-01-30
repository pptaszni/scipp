/// @file
/// SPDX-License-Identifier: GPL-3.0-or-later
/// @author Simon Heybrock
/// Copyright &copy; 2019 ISIS Rutherford Appleton Laboratory, NScD Oak Ridge
/// National Laboratory, and European Spallation Source ERIC.
#include "convert.h"
#include "dataset.h"
#include "md_zip_view.h"

namespace neutron {
namespace tof {
Dataset tofToEnergy(const Dataset &d) {
  // TODO Could in principle also support inelastic. Note that the conversion in
  // Mantid is wrong since it handles inelastic data as if it were elastic.
  if (d.contains(Coord::Ei{}) || d.contains(Coord::Ef{}))
    throw std::runtime_error("Dataset contains Coord::Ei or Coord::Ef. "
                             "However, conversion to Dim::Energy is currently "
                             "only supported for elastic scattering.");

  // 1. Compute conversion factor
  const auto &compPos =
      d.get<const Coord::ComponentInfo>()[0].get<const Coord::Position>();
  // TODO Need a better mechanism to identify source and sample.
  const auto &sourcePos = compPos[0];
  const auto &samplePos = compPos[1];
  const double l1 = (sourcePos - samplePos).norm();

  const auto &dims = d.contains(Coord::Position{})
                         ? d(Coord::Position{}).dimensions()
                         : d(Coord::DetectorGrouping{}).dimensions();
  Variable conversionFactor(Data::Value{}, dims);

  MDZipView<const Coord::Position> specPos(d);
  const auto factor = [&](const auto &item) {
    const auto &pos = item.template get<Coord::Position>();
    double l_total = l1 + (samplePos - pos).norm();
    // TODO Up to physical constants.
    return l_total * l_total;
  };
  // TODO Must also update unit of conversionFactor.
  std::transform(specPos.begin(), specPos.end(),
                 conversionFactor.get<Data::Value>().begin(), factor);

  // 2. Transform variables
  Dataset converted;
  for (const auto &var : d) {
    auto varDims = var.dimensions();
    if (varDims.contains(Dim::Tof))
      varDims.relabel(varDims.index(Dim::Tof), Dim::Energy);
    if (var.tag() == Coord::Tof{}) {
      // TODO Need to extend the broadcasting capabilities to broadcast to the
      // union of dimensions of both operands in a binary operation.
      auto dims = conversionFactor.dimensions();
      for (const Dim dim : varDims.labels())
        if (!dims.contains(dim))
          dims.addInner(dim, varDims[dim]);
      // TODO Should have a broadcasting assign method?
      Variable energy(Coord::Energy{}, dims, dims.volume(), 1.0);
      energy *= conversionFactor;
      // The reshape is just to remap the dimension label, should probably do
      // this differently.
      energy /= (var * var).reshape(varDims);
      converted.insert(energy);
    } else if (var.tag() == Data::Events{}) {
      throw std::runtime_error(
          "TODO Converting units of event data not implemented yet.");
    } else {
      // TODO Changing Dim::Tof to Dim::Energy. Currently this is not possible
      // without making a copy of the data, which is inefficient. If we cannot
      // move the dimensions outside the cow_ptr in Variable, another option
      // would be to support in-place modification. Probably a better solution
      // would be to decouple the pointer holding VariableConcept from the COW
      // mechanism, i.e., to COW inside VariableConcept to hold the data
      // array?
      // TODO Also need to check here if variable contains count/bin_width,
      // should fail then?
      converted.insert(var.reshape(varDims));
    }
  }

  return converted;
}

Dataset tofToDeltaE(const Dataset &d) {
  // TODO Units and physical constants!

  // There are two cases, direct inelastic and indirect inelastic. We can
  // distinguish them by the content of d.
  if (d.contains(Coord::Ei{}) && d.contains(Coord::Ef{}))
    throw std::runtime_error("Dataset contains Coord::Ei as well as Coord::Ef, "
                             "cannot have both for inelastic scattering.");

  // 1. Compute conversion factors
  const auto &compPos =
      d.get<const Coord::ComponentInfo>()[0].get<const Coord::Position>();
  const auto &sourcePos = compPos[0];
  const auto &samplePos = compPos[1];
  const double l1 = (sourcePos - samplePos).norm();

  const auto &dims = d.contains(Coord::Position{})
                         ? d(Coord::Position{}).dimensions()
                         : d(Coord::DetectorGrouping{}).dimensions();
  Variable tofShift(Data::Value{}, {});
  Variable scale(Data::Value{}, {});

  if (d.contains(Coord::Ei{})) {
    // Direct-inelastic.

    // This is how we support multi-Ei data!
    tofShift.setDimensions(d(Coord::Ei{}).dimensions());
    const auto &Ei = d.get<const Coord::Ei>();
    std::transform(Ei.begin(), Ei.end(), tofShift.get<Data::Value>().begin(),
                   [&l1](const double Ei) { return l1 / sqrt(Ei); });

    scale.setDimensions(dims);
    MDZipView<const Coord::Position> specPos(d);
    std::transform(specPos.begin(), specPos.end(),
                   scale.get<Data::Value>().begin(), [&](const auto &item) {
                     const auto &pos = item.template get<Coord::Position>();
                     const double l2 = (samplePos - pos).norm();
                     return l2 * l2;
                   });
  } else if (d.contains(Coord::Ef{})) {
    // Indirect-inelastic.

    tofShift.setDimensions(dims);
    // Ef can be different for every spectrum so we access it also via a view.
    MDZipView<const Coord::Position, const Coord::Ef> geometry(d);
    std::transform(geometry.begin(), geometry.end(),
                   tofShift.get<Data::Value>().begin(), [&](const auto &item) {
                     const auto &pos = item.template get<Coord::Position>();
                     const auto &Ef = item.template get<Coord::Ef>();
                     const double l2 = (samplePos - pos).norm();
                     return l2 * l2 / sqrt(Ef);
                   });

    scale.setDimensions({});
    scale.get<Data::Value>()[0] = l1 * l1;
  } else {
    throw std::runtime_error("Dataset contains neither Coord::Ei nor "
                             "Coord::Ef, this does not look like "
                             "inelastic-scattering data.");
  }

  // 2. Transform variables
  Dataset converted;
  for (const auto &var : d) {
    auto varDims = var.dimensions();
    if (varDims.contains(Dim::Tof))
      varDims.relabel(varDims.index(Dim::Tof), Dim::DeltaE);
    if (var.tag() == Coord::Tof{}) {
      auto dims = scale.dimensions();
      for (const Dim dim : varDims.labels())
        if (!dims.contains(dim))
          dims.addInner(dim, varDims[dim]);
      Variable E(Coord::DeltaE{}, dims, dims.volume(), 1.0);
      E *= var.reshape(varDims);
      E -= tofShift;
      E *= E;
      E = 1.0 / std::move(E);
      E *= scale;
      if (d.contains(Coord::Ei{})) {
        converted.insert(-(std::move(E) - d(Coord::Ei{})));
      } else {
        converted.insert(std::move(E) - d(Coord::Ef{}));
      }
    } else if (var.tag() == Data::Events{}) {
      throw std::runtime_error(
          "TODO Converting units of event data not implemented yet.");
    } else {
      converted.insert(var.reshape(varDims));
    }
  }

  return converted;
}
} // namespace tof
} // namespace neutron

Dataset convert(const Dataset &d, const Dim from, const Dim to) {
  if ((from == Dim::Tof) && (to == Dim::Energy))
    return neutron::tof::tofToEnergy(d);
  if ((from == Dim::Tof) && (to == Dim::DeltaE))
    return neutron::tof::tofToDeltaE(d);
  throw std::runtime_error(
      "Conversion between requested dimensions not implemented yet.");
  // How to convert? There are several cases:
  // 1. Tof conversion as Mantid's ConvertUnits.
  // 2. Axis conversion as Mantid's ConvertSpectrumAxis.
  // 3. Conversion of multiple dimensions simultaneuously, e.g., to Q, which
  //    cannot be done here since it affects more than one input and output
  //    dimension. Should we have a variant that accepts a list of dimensions
  //    for input and output?
  // 4. Conversion from 1 to N or N to 1, e.g., Dim::Spectrum to X and Y pixel
  //    index.
  // Can Dim::Spectrum be converted to anything? Should we require a matching
  // coordinate when doing a conversion? This does not make sense:
  // auto converted = convert(dataset, Dim::Spectrum, Dim::Tof);
  // This does if we can lookup the TwoTheta, make axis here, or require it?
  // Should it do the reordering? Is sorting separately much less efficient?
  // Dim::Spectrum is discrete, Dim::TwoTheta is in principle contiguous. How to
  // handle that? Do we simply want to sort instead? Discrete->contiguous can be
  // handled by binning? Or is Dim::TwoTheta implicitly also discrete?
  // auto converted = convert(dataset, Dim::Spectrum, Dim::TwoTheta);
  // This is a *derived* coordinate, no need to store it explicitly? May even be
  // prevented?
  // MDZipView<const Coord::TwoTheta>(dataset);
}
