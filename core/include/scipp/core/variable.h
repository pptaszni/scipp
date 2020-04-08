// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef SCIPP_CORE_VARIABLE_H
#define SCIPP_CORE_VARIABLE_H

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "scipp-core_export.h"
#include "scipp/common/index.h"
#include "scipp/common/span.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/dtype.h"
#include "scipp/core/element_array.h"
#include "scipp/core/element_array_view.h"
#include "scipp/core/slice.h"
#include "scipp/core/tag_util.h"
#include "scipp/core/variable_keyword_arg_constructor.h"
#include "scipp/units/unit.h"

namespace scipp::dataset {
class DatasetConstView;
class DatasetView;
class Dataset;
class DataArray;
class DataArrayView;
class DataArrayConstView;
template <class T> typename T::view_type makeViewItem(T &);
} // namespace scipp::dataset

namespace scipp::core {

namespace detail {
template <class T> struct element_type { using type = T; };
template <class T> struct element_type<sparse_container<T>> { using type = T; };
template <class T> struct element_type<const sparse_container<T>> {
  using type = T;
};
template <class T> using element_type_t = typename element_type<T>::type;

std::vector<scipp::index> reorderedShape(const scipp::span<const Dim> &order,
                                         const Dimensions &dimensions);
void expect0D(const Dimensions &dims);
} // namespace detail

template <class T> struct is_sparse_container : std::false_type {};
template <class T>
struct is_sparse_container<sparse_container<T>> : std::true_type {};

class Variable;
template <class... Known> class VariableConceptHandle_impl;
// Any item type that is listed here explicitly can be used with the templated
// `transform`, i.e., we can pass arbitrary functors/lambdas to process data.
#define KNOWN                                                                  \
  double, float, int64_t, int32_t, bool, Eigen::Vector3d,                      \
      sparse_container<double>, sparse_container<float>,                       \
      sparse_container<int64_t>, sparse_container<int32_t>,                    \
      sparse_container<bool>, span<const double>, span<double>,                \
      span<const float>, span<float>, span<const int64_t>, span<int64_t>,      \
      span<const int32_t>, span<int32_t>
using VariableConceptHandle = VariableConceptHandle_impl<KNOWN>;

/// Abstract base class for any data that can be held by Variable. Also used to
/// hold views to data by (Const)VariableView. This is using so-called
/// concept-based polymorphism, see talks by Sean Parent.
///
/// This is the most generic representation for a multi-dimensional array of
/// data. More operations are supportd by the partially-typed VariableConceptT.
class SCIPP_CORE_EXPORT VariableConcept {
public:
  VariableConcept(const Dimensions &dimensions);
  virtual ~VariableConcept() = default;

  virtual DType dtype() const noexcept = 0;
  virtual VariableConceptHandle clone() const = 0;
  virtual VariableConceptHandle
  makeDefaultFromParent(const Dimensions &dims) const = 0;
  virtual VariableConceptHandle makeView() const = 0;
  virtual VariableConceptHandle makeView() = 0;
  virtual VariableConceptHandle makeView(const Dim dim,
                                         const scipp::index begin,
                                         const scipp::index end = -1) const = 0;
  virtual VariableConceptHandle makeView(const Dim dim,
                                         const scipp::index begin,
                                         const scipp::index end = -1) = 0;

  virtual VariableConceptHandle reshape(const Dimensions &dims) const = 0;
  virtual VariableConceptHandle reshape(const Dimensions &dims) = 0;

  virtual VariableConceptHandle
  transpose(const std::vector<Dim> &dms) const = 0;
  virtual VariableConceptHandle transpose(const std::vector<Dim> &dms) = 0;

  virtual bool operator==(const VariableConcept &other) const = 0;
  virtual bool isSame(const VariableConcept &other) const = 0;

  virtual bool isContiguous() const = 0;
  virtual bool isView() const = 0;
  virtual bool isConstView() const = 0;
  virtual bool hasVariances() const noexcept = 0;

  virtual scipp::index size() const = 0;
  virtual void copy(const VariableConcept &other, const Dim dim,
                    const scipp::index offset, const scipp::index otherBegin,
                    const scipp::index otherEnd) = 0;

  virtual void setVariances(Variable &&variances) = 0;

  const Dimensions &dims() const { return m_dimensions; }

  friend class Variable;

private:
  Dimensions m_dimensions;
};

/// Partially typed implementation of VariableConcept. This is a common base
/// class for DataModel<T> and ViewModel<T>. The former holds data in a
/// contiguous array, whereas the latter is a (potentially non-contiguous) view
/// into the former. This base class implements functionality that is common to
/// both, for a specific T.
template <class T> class VariableConceptT : public VariableConcept {
public:
  using value_type = T;

  VariableConceptT(const Dimensions &dimensions)
      : VariableConcept(dimensions) {}

  DType dtype() const noexcept override { return scipp::core::dtype<T>; }
  static DType static_dtype() noexcept { return scipp::core::dtype<T>; }

  virtual scipp::span<T> values() = 0;
  virtual scipp::span<T> values(const Dim dim, const scipp::index begin,
                                const scipp::index end) = 0;
  virtual scipp::span<const T> values() const = 0;
  virtual scipp::span<const T> values(const Dim dim, const scipp::index begin,
                                      const scipp::index end) const = 0;
  virtual scipp::span<T> variances() = 0;
  virtual scipp::span<T> variances(const Dim dim, const scipp::index begin,
                                   const scipp::index end) = 0;
  virtual scipp::span<const T> variances() const = 0;
  virtual scipp::span<const T> variances(const Dim dim,
                                         const scipp::index begin,
                                         const scipp::index end) const = 0;
  virtual ElementArrayView<T> valuesView(const Dimensions &dims) = 0;
  virtual ElementArrayView<T> valuesView(const Dimensions &dims, const Dim dim,
                                         const scipp::index begin) = 0;
  virtual ElementArrayView<const T>
  valuesView(const Dimensions &dims) const = 0;
  virtual ElementArrayView<const T>
  valuesView(const Dimensions &dims, const Dim dim,
             const scipp::index begin) const = 0;
  virtual ElementArrayView<T> variancesView(const Dimensions &dims) = 0;
  virtual ElementArrayView<T> variancesView(const Dimensions &dims,
                                            const Dim dim,
                                            const scipp::index begin) = 0;
  virtual ElementArrayView<const T>
  variancesView(const Dimensions &dims) const = 0;
  virtual ElementArrayView<const T>
  variancesView(const Dimensions &dims, const Dim dim,
                const scipp::index begin) const = 0;
  virtual ElementArrayView<const T>
  valuesReshaped(const Dimensions &dims) const = 0;
  virtual ElementArrayView<T> valuesReshaped(const Dimensions &dims) = 0;
  virtual ElementArrayView<const T>
  variancesReshaped(const Dimensions &dims) const = 0;
  virtual ElementArrayView<T> variancesReshaped(const Dimensions &dims) = 0;

  virtual std::unique_ptr<VariableConceptT> copyT() const = 0;

  VariableConceptHandle
  makeDefaultFromParent(const Dimensions &dims) const override;

  VariableConceptHandle makeView() const override;

  VariableConceptHandle makeView() override;

  VariableConceptHandle makeView(const Dim dim, const scipp::index begin,
                                 const scipp::index end) const override;

  VariableConceptHandle makeView(const Dim dim, const scipp::index begin,
                                 const scipp::index end) override;

  VariableConceptHandle reshape(const Dimensions &dims) const override;

  VariableConceptHandle reshape(const Dimensions &dims) override;

  VariableConceptHandle transpose(const std::vector<Dim> &dims) const override;

  VariableConceptHandle transpose(const std::vector<Dim> &dims) override;

  bool operator==(const VariableConcept &other) const override;
  void copy(const VariableConcept &other, const Dim dim,
            const scipp::index offset, const scipp::index otherBegin,
            const scipp::index otherEnd) override;
};

template <class... Known> class VariableConceptHandle_impl {
public:
  using variant_t =
      std::variant<const VariableConcept *, const VariableConceptT<Known> *...>;

  VariableConceptHandle_impl()
      : m_object(std::unique_ptr<VariableConcept>(nullptr)) {}
  template <class T> VariableConceptHandle_impl(T object) {
    using value_t = typename T::element_type::value_type;
    if constexpr ((std::is_same_v<value_t, Known> || ...))
      m_object = std::unique_ptr<VariableConceptT<value_t>>(std::move(object));
    else
      m_object = std::unique_ptr<VariableConcept>(std::move(object));
  }
  VariableConceptHandle_impl(VariableConceptHandle_impl &&) = default;
  VariableConceptHandle_impl(const VariableConceptHandle_impl &other)
      : VariableConceptHandle_impl(other ? other->clone()
                                         : VariableConceptHandle_impl()) {}
  VariableConceptHandle_impl &
  operator=(VariableConceptHandle_impl &&) = default;
  VariableConceptHandle_impl &
  operator=(const VariableConceptHandle_impl &other) {
    if (*this && other) {
      // Avoid allocation of new element_array if output is of correct shape.
      // This yields a 5x speedup in assignment operations of variables.
      auto &concept = **this;
      auto &otherConcept = *other;
      if (!concept.isView() && !otherConcept.isView() &&
          concept.dtype() == otherConcept.dtype() &&
          concept.dims() == otherConcept.dims() &&
          concept.hasVariances() == otherConcept.hasVariances()) {
        concept.copy(otherConcept, Dim::Invalid, 0, 0, 1);
        return *this;
      }
    }
    return *this = other ? other->clone() : VariableConceptHandle_impl();
  }

  explicit operator bool() const noexcept;
  VariableConcept &operator*() const;
  VariableConcept *operator->() const;

  const auto &mutableVariant() const noexcept { return m_object; }

  variant_t variant() const noexcept;

private:
  std::variant<std::unique_ptr<VariableConcept>,
               std::unique_ptr<VariableConceptT<Known>>...>
      m_object;
};

class VariableConstView;
class VariableView;

template <class T> constexpr bool is_variable_or_view() {
  return std::is_same_v<T, Variable> || std::is_same_v<T, VariableConstView> ||
         std::is_same_v<T, VariableView>;
}

template <class T> constexpr bool is_container_or_view() {
  return std::is_same_v<T, dataset::Dataset> ||
         std::is_same_v<T, dataset::DatasetView> ||
         std::is_same_v<T, dataset::DatasetConstView> ||
         std::is_same_v<T, Variable> || std::is_same_v<T, VariableView> ||
         std::is_same_v<T, VariableConstView> ||
         std::is_same_v<T, dataset::DataArray> ||
         std::is_same_v<T, dataset::DataArrayView> ||
         std::is_same_v<T, dataset::DataArrayConstView>;
}

namespace detail {
template <class T> struct default_init {
  static T value() { return T(); }
};
// Eigen does not zero-initialize matrices (vectors), which is a recurrent
// source of bugs. Variable does zero-init instead.
template <class T, int Rows, int Cols>
struct default_init<Eigen::Matrix<T, Rows, Cols>> {
  static Eigen::Matrix<T, Rows, Cols> value() {
    return Eigen::Matrix<T, Rows, Cols>::Zero();
  }
};
} // namespace detail

template <class T, class... Ts> Variable makeVariable(Ts &&... ts);

/// Variable is a type-erased handle to any data structure representing a
/// multi-dimensional array. It has a name, a unit, and a set of named
/// dimensions.
class SCIPP_CORE_EXPORT Variable {
public:
  using const_view_type = VariableConstView;
  using view_type = VariableView;

  Variable() = default;
  // Having this non-explicit is convenient when passing (potential)
  // variable slices to functions that do not support slices, but implicit
  // conversion may introduce risks, so there is a trade-of here.
  explicit Variable(const VariableConstView &slice);
  Variable(const Variable &parent, const Dimensions &dims);
  Variable(const VariableConstView &parent, const Dimensions &dims);
  Variable(const Variable &parent, VariableConceptHandle data);
  template <class T>
  Variable(const units::Unit unit, const Dimensions &dimensions, T object);
  template <class T>
  Variable(const units::Unit unit, const Dimensions &dimensions, T values,
           T variances);

  template <class T>
  static Variable create(const units::Unit &u, const Dims &d, const Shape &s,
                         std::optional<detail::element_array<T>> &&val,
                         std::optional<detail::element_array<T>> &&var);

  template <class T>
  static Variable create(const units::Unit &u, const Dimensions &d,
                         std::optional<detail::element_array<T>> &&val,
                         std::optional<detail::element_array<T>> &&var);

  template <class T>
  Variable(const Dimensions &dimensions, std::initializer_list<T> values_)
      : Variable(units::dimensionless, std::move(dimensions),
                 detail::element_array<T>(values_.begin(), values_.end())) {}

  /// This is used to provide the constructors:
  /// Variable(DType, Dims, Shape, Unit, Values<T1>, Variances<T2>)
  /// with the only one obligatory argument DType, the other arguments are not
  /// obligatory and could be given in arbitrary order. Example:
  /// Variable(dtype<float>, units::Unit(units::kg), Shape{1, 2}, Dims{Dim::X,
  /// Dim::Y}, Values({3, 4})).
  template <class... Ts> Variable(const DType &type, Ts &&... args);

  explicit operator bool() const noexcept { return m_object.operator bool(); }
  Variable operator~() const;

  units::Unit unit() const { return m_unit; }
  void setUnit(const units::Unit &unit) { m_unit = unit; }
  constexpr void expectCanSetUnit(const units::Unit &) const noexcept {}

  Dimensions dims() const && { return m_object->dims(); }
  const Dimensions &dims() const & { return m_object->dims(); }
  void setDims(const Dimensions &dimensions);

  DType dtype() const noexcept { return data().dtype(); }

  bool hasVariances() const noexcept { return data().hasVariances(); }

  template <class T> auto values() const { return scipp::span(cast<T>()); }
  template <class T> auto values() { return scipp::span(cast<T>()); }
  template <class T> auto variances() const {
    return scipp::span(cast<T>(true));
  }
  template <class T> auto variances() { return scipp::span(cast<T>(true)); }
  template <class T> const auto &value() const {
    detail::expect0D(dims());
    return values<T>()[0];
  }
  template <class T> const auto &variance() const {
    detail::expect0D(dims());
    return variances<T>()[0];
  }
  template <class T> auto &value() {
    detail::expect0D(dims());
    return values<T>()[0];
  }
  template <class T> auto &variance() {
    detail::expect0D(dims());
    return variances<T>()[0];
  }

  // ATTENTION: It is really important to avoid any function returning a
  // (Const)VariableView for rvalue Variable. Otherwise the resulting slice
  // will point to free'ed memory.
  VariableConstView slice(const Slice slice) const &;
  Variable slice(const Slice slice) const &&;
  VariableView slice(const Slice slice) &;
  Variable slice(const Slice slice) &&;

  VariableConstView reshape(const Dimensions &dims) const &;
  VariableView reshape(const Dimensions &dims) &;
  // Note: Do we have to delete the `const &&` version? Consider
  //   const Variable var;
  //   std::move(var).reshape({});
  // This calls `reshape() const &` but in this case it is not a temporary and
  // will not go out of scope, so that is ok (unless someone changes var and
  // expects the reshaped view to be still valid).
  Variable reshape(const Dimensions &dims) &&;

  VariableConstView transpose(const std::vector<Dim> &dims = {}) const &;
  VariableView transpose(const std::vector<Dim> &dims = {}) &;
  // Note: the same issue as for reshape above
  Variable transpose(const std::vector<Dim> &dims = {}) &&;
  void rename(const Dim from, const Dim to);

  bool operator==(const VariableConstView &other) const;
  bool operator!=(const VariableConstView &other) const;
  Variable operator-() const;

  Variable &operator+=(const VariableConstView &other) &;
  template <typename T, typename = std::enable_if_t<!is_variable_or_view<T>()>>
  Variable &operator+=(const T v) & {
    return *this += makeVariable<T>(Values{v});
  }

  Variable &operator-=(const VariableConstView &other) &;
  template <typename T, typename = std::enable_if_t<!is_variable_or_view<T>()>>
  Variable &operator-=(const T v) & {
    return *this -= makeVariable<T>(Values{v});
  }

  Variable &operator*=(const VariableConstView &other) &;
  template <typename T, typename = std::enable_if_t<!is_variable_or_view<T>()>>
  Variable &operator*=(const T v) & {
    return *this *= makeVariable<T>(Values{v});
  }
  template <class T>
  Variable &operator*=(const boost::units::quantity<T> &quantity) & {
    setUnit(unit() * units::Unit(T{}));
    return *this *= quantity.value();
  }

  Variable &operator/=(const VariableConstView &other) &;
  template <typename T, typename = std::enable_if_t<!is_variable_or_view<T>()>>
  Variable &operator/=(const T v) & {
    return *this /= makeVariable<T>(Values{v});
  }
  template <class T>
  Variable &operator/=(const boost::units::quantity<T> &quantity) & {
    setUnit(unit() / units::Unit(T{}));
    return *this /= quantity.value();
  }

  Variable &operator|=(const VariableConstView &other) &;
  Variable &operator&=(const VariableConstView &other) &;
  Variable &operator^=(const VariableConstView &other) &;

  const VariableConcept &data() const && = delete;
  const VariableConcept &data() const & { return *m_object; }
  VariableConcept &data() && = delete;
  VariableConcept &data() & { return *m_object; }

  /// Return variant of pointers to underlying data.
  ///
  /// This is intended for internal use (such as implementing transform
  /// algorithms) and should not need to be used directly by higher-level code.
  auto dataHandle() const && = delete;
  auto dataHandle() const & { return m_object.variant(); }
  const auto &dataHandle() && = delete;
  const auto &dataHandle() & { return m_object.mutableVariant(); }

  void setVariances(Variable v);

private:
  template <class... Ts> struct ConstructVariable {
    template <class T> struct Maker { static Variable apply(Ts &&... args); };

    static Variable make(Ts &&... args, DType type);
  };

  template <class T>
  const detail::element_array<T> &cast(const bool variances = false) const;
  template <class T>
  detail::element_array<T> &cast(const bool variances = false);

  units::Unit m_unit;
  VariableConceptHandle m_object;
};
/// This is used to provide the fabric function:
/// makeVariable<type>(Dims, Shape, Unit, Values<T1>, Variances<T2>)
/// with the obligatory template argument type, function arguments are not
/// obligatory and could be given in arbitrary order, but suposed to be wrapped
/// in certain classes.
/// Example: makeVariable<float>(units::Unit(units::kg),
/// Shape{1, 2}, Dims{Dim::X, Dim::Y}, Values({3, 4})).
template <class T, class... Ts> Variable makeVariable(Ts &&... ts) {
  static_assert(!std::disjunction_v<std::is_lvalue_reference<Ts>...>,
                "makeVariable inputs must be r-value references");
  using helper = detail::ConstructorArgumentsMatcher<Variable, Ts...>;
  constexpr bool useDimsAndShape =
      helper::template checkArgTypesValid<units::Unit, Dims, Shape>();
  constexpr bool useDimensions =
      helper::template checkArgTypesValid<units::Unit, Dimensions>();

  static_assert(
      useDimsAndShape || useDimensions,
      "Arguments: units::Unit, Shape, Dims, Values and Variances could only "
      "be used. Example: Variable(dtype<float>, units::Unit(units::kg), "
      "Shape{1, 2}, Dims{Dim::X, Dim::Y}, Values({3, 4}))");

  if constexpr (useDimsAndShape) {
    auto [valArgs, varArgs, nonData] =
        helper::template extractArguments<units::Unit, Dims, Shape>(
            std::forward<Ts>(ts)...);
    return helper::template construct<T>(std::move(valArgs), std::move(varArgs),
                                         std::move(nonData));
  } else {
    auto [valArgs, varArgs, nonData] =
        helper::template extractArguments<units::Unit, Dimensions>(
            std::forward<Ts>(ts)...);
    return helper::template construct<T>(std::move(valArgs), std::move(varArgs),
                                         std::move(nonData));
  }
}

namespace detail {
template <class T>
Variable from_dimensions_and_unit(const Dimensions &dms, const units::Unit &u);

template <class T>
Variable from_dimensions_and_unit_with_variances(const Dimensions &dms,
                                                 const units::Unit &u);
} // namespace detail

template <class... Ts>
template <class T>
Variable Variable::ConstructVariable<Ts...>::Maker<T>::apply(Ts &&... ts) {
  return makeVariable<T>(std::forward<Ts>(ts)...);
}

template <class... Ts>
Variable Variable::ConstructVariable<Ts...>::make(Ts &&... args, DType type) {
  return CallDType<
      double, float, int64_t, int32_t, bool, Eigen::Vector3d, std::string,
      event_list<double>, event_list<float>, event_list<int64_t>,
      event_list<int32_t>>::apply<Maker>(type, std::forward<Ts>(args)...);
}

template <class... Ts>
Variable::Variable(const DType &type, Ts &&... args)
    : Variable{
          ConstructVariable<Ts...>::make(std::forward<Ts>(args)..., type)} {}

/// Non-mutable view into (a subset of) a Variable.
class SCIPP_CORE_EXPORT VariableConstView {
public:
  using value_type = Variable;

  VariableConstView() = default;
  VariableConstView(const Variable &variable) : m_variable(&variable) {}
  VariableConstView(const Variable &variable, const Dimensions &dims)
      : m_variable(&variable), m_view(variable.data().reshape(dims)) {}
  VariableConstView(const Variable &variable, const Dim dim,
                    const scipp::index begin, const scipp::index end = -1)
      : m_variable(&variable),
        m_view(variable.data().makeView(dim, begin, end)) {}
  VariableConstView(const VariableConstView &slice, const Dim dim,
                    const scipp::index begin, const scipp::index end = -1)
      : m_variable(slice.m_variable),
        m_view(slice.data().makeView(dim, begin, end)) {}

  explicit operator bool() const noexcept {
    return m_variable && m_variable->operator bool();
  }

  auto operator~() const { return m_variable->operator~(); }

  VariableConstView slice(const Slice slice) const {
    return VariableConstView(*this, slice.dim(), slice.begin(), slice.end());
  }

  VariableConstView transpose(const std::vector<Dim> &dims = {}) const;
  // Note the return type. Reshaping a non-contiguous slice cannot return a
  // slice in general so we must return a copy of the data.
  Variable reshape(const Dimensions &dims) const;

  units::Unit unit() const { return m_variable->unit(); }

  // Note: Returning by value to avoid issues with referencing a temporary
  // (VariableView is returned by-value from DatasetSlice).
  Dimensions dims() const {
    if (m_view)
      return m_view->dims();
    else
      return m_variable->dims();
  }

  std::vector<scipp::index> strides() const {
    const auto parent = m_variable->dims();
    std::vector<scipp::index> strides;
    for (const auto &label : parent.labels())
      if (dims().contains(label))
        strides.emplace_back(parent.offset(label));
    return strides;
  }

  DType dtype() const noexcept { return m_variable->dtype(); }

  const VariableConcept &data() const && = delete;
  const VariableConcept &data() const & {
    if (m_view)
      return *m_view;
    else
      return m_variable->data();
  }

  auto dataHandle() const && = delete;
  auto dataHandle() const & {
    if (m_view)
      return m_view.variant();
    else
      return m_variable->dataHandle();
  }

  bool hasVariances() const noexcept { return m_variable->hasVariances(); }

  // Note: This return a view object (a ElementArrayView) that does reference
  // members owner by *this. Therefore we can support this even for
  // temporaries and we do not need to delete the rvalue overload, unlike for
  // many other methods. The data is owned by the underlying variable so it
  // will not be deleted even if *this is a temporary and gets deleted.
  template <class T> auto values() const { return cast<T>(); }
  template <class T> auto variances() const { return castVariances<T>(); }
  template <class T> const auto &value() const {
    detail::expect0D(dims());
    return values<T>()[0];
  }
  template <class T> const auto &variance() const {
    detail::expect0D(dims());
    return variances<T>()[0];
  }

  bool operator==(const VariableConstView &other) const;
  bool operator!=(const VariableConstView &other) const;
  Variable operator-() const;

  auto &underlying() const { return *m_variable; }

private:
  template <class Var>
  static VariableConstView makeTransposed(Var &var,
                                          const std::vector<Dim> &dimOrder) {
    auto res = VariableConstView(var);
    res.m_view = res.data().transpose(dimOrder);
    return res;
  }

protected:
  friend class Variable;

  template <class T> const ElementArrayView<const T> cast() const;
  template <class T> const ElementArrayView<const T> castVariances() const;

  const Variable *m_variable{nullptr};
  VariableConceptHandle m_view;
};

/** Mutable view into (a subset of) a Variable.
 *
 * By inheriting from VariableConstView any code that works for
 * VariableConstView will automatically work also for this mutable variant.*/
class SCIPP_CORE_EXPORT VariableView : public VariableConstView {
public:
  VariableView() = default;
  VariableView(Variable &variable)
      : VariableConstView(variable), m_mutableVariable(&variable) {}
  // Note that we use the basic constructor of VariableConstView to avoid
  // creation of a const m_view, which would be overwritten immediately.
  VariableView(Variable &variable, const Dimensions &dims)
      : VariableConstView(variable), m_mutableVariable(&variable) {
    m_view = variable.data().reshape(dims);
  }
  VariableView(Variable &variable, const Dim dim, const scipp::index begin,
               const scipp::index end = -1)
      : VariableConstView(variable), m_mutableVariable(&variable) {
    m_view = variable.data().makeView(dim, begin, end);
  }
  VariableView(const VariableView &slice, const Dim dim,
               const scipp::index begin, const scipp::index end = -1)
      : VariableConstView(slice), m_mutableVariable(slice.m_mutableVariable) {
    m_view = slice.data().makeView(dim, begin, end);
  }

  VariableView slice(const Slice slice) const {
    return VariableView(*this, slice.dim(), slice.begin(), slice.end());
  }

  VariableView transpose(const std::vector<Dim> &dims = {}) const;

  using VariableConstView::data;

  VariableConcept &data() const && = delete;
  VariableConcept &data() const & {
    if (!m_view)
      return m_mutableVariable->data();
    return *m_view;
  }

  const auto &dataHandle() const && = delete;
  const auto &dataHandle() const & {
    if (!m_view)
      return m_mutableVariable->dataHandle();
    return m_view.mutableVariant();
  }

  // Note: No need to delete rvalue overloads here, see VariableConstView.
  template <class T> auto values() const { return cast<T>(); }
  template <class T> auto variances() const { return castVariances<T>(); }
  template <class T> auto &value() const {
    detail::expect0D(dims());
    return values<T>()[0];
  }
  template <class T> auto &variance() const {
    detail::expect0D(dims());
    return variances<T>()[0];
  }

  // Note: We want to support things like `var(Dim::X, 0) += var2`, i.e., when
  // the left-hand-side is a temporary. This is ok since data is modified in
  // underlying Variable. However, we do not return the typical `VariableView
  // &` from these operations since that could reference a temporary. Due to the
  // way Python implements things like __iadd__ we must return an object
  // referencing the data though. We therefore return by value (this is not for
  // free since it involves a memory allocation but is probably relatively cheap
  // compared to other things). If the return by value turns out to be a
  // performance issue, another option is to have overloads for *this of types
  // `&` and `&&` with distinct return types (by reference in the first case, by
  // value in the second). In principle we may also change the implementation of
  // the Python exports to return `a` after calling `a += b` instead of
  // returning `a += b` but I am not sure how Pybind11 handles object lifetimes
  // (would this suffer from the same issue?).
  template <class T> VariableView assign(const T &other) const;

  VariableView operator+=(const VariableConstView &other) const;
  template <typename T, typename = std::enable_if_t<!is_variable_or_view<T>()>>
  VariableView operator+=(const T v) const {
    return *this += makeVariable<T>(Values{v});
  }

  VariableView operator-=(const VariableConstView &other) const;
  template <typename T, typename = std::enable_if_t<!is_variable_or_view<T>()>>
  VariableView operator-=(const T v) const {
    return *this -= makeVariable<T>(Values{v});
  }

  VariableView operator*=(const VariableConstView &other) const;
  template <typename T, typename = std::enable_if_t<!is_variable_or_view<T>()>>
  VariableView operator*=(const T v) const {
    return *this *= makeVariable<T>(Values{v});
  }

  VariableView operator/=(const VariableConstView &other) const;
  template <typename T, typename = std::enable_if_t<!is_variable_or_view<T>()>>
  VariableView operator/=(const T v) const {
    return *this /= makeVariable<T>(Values{v});
  }

  VariableView operator|=(const VariableConstView &other) const;
  VariableView operator&=(const VariableConstView &other) const;
  VariableView operator^=(const VariableConstView &other) const;

  void setVariances(Variable v) const;

  void setUnit(const units::Unit &unit) const;
  void expectCanSetUnit(const units::Unit &unit) const;
  scipp::index size() const { return data().size(); }

private:
  friend class Variable;
  friend class dataset::DataArrayConstView;
  template <class T> friend typename T::view_type dataset::makeViewItem(T &);

  // For internal use in DataArrayConstView.
  explicit VariableView(VariableConstView &&base)
      : VariableConstView(std::move(base)), m_mutableVariable{nullptr} {}

  template <class Var>
  static VariableView makeTransposed(Var &var,
                                     const std::vector<Dim> &dimOrder) {
    auto res = VariableView(var);
    res.m_view = res.data().transpose(dimOrder);
    return res;
  }

  template <class T> ElementArrayView<T> cast() const;
  template <class T> ElementArrayView<T> castVariances() const;

  Variable *m_mutableVariable{nullptr};
};

SCIPP_CORE_EXPORT Variable copy(const VariableConstView &var);

SCIPP_CORE_EXPORT bool is_events(const VariableConstView &var);

} // namespace scipp::core

namespace scipp {
using core::Dimensions;
using core::Dims;
using core::DType;
using core::dtype;
using core::event_list;
using core::makeVariable;
using core::Shape;
using core::Slice;
using core::Values;
using core::Variable;
using core::VariableConstView;
using core::VariableView;
using core::Variances;
} // namespace scipp

#endif // SCIPP_CORE_VARIABLE_H
