Transforming data
=================

Overview
--------

Scipp can apply custom operations to the elements of one or more variables.
This is essentially are more advanced version of ``std::transform``.

Two alternatives are provided:

- ``transform_in_place`` transforms its first argument.
- ``transform`` returns a new ``Variable``.

Both variants support:

- Automatic broadcasting and alignment based on dimension labels.
  This does also include event data, and operations mixing events and dense data are supported.
- Automatic propagation of uncertainties, provided that the user-provided operation is built from known existing operations.
- Operations between different data types, and operations that produce outputs with a new data type.
- Handling of units.

Basic usage
-----------

Transform requires:

- A list of types (or type-combinations) to support.
  Code is only instantiated for variables of those types.
- A functor, typically a lambda or an "overloaded" lambda.
  This can also be used to pass special flags, e.g., for disabling code generation for arguments that have variances, generating a runtime failure instead.

Example 1
~~~~~~~~~

Transform two variables with two type combinations:

- ``a`` of type ``double`` and ``b`` of type ``float``
- ``a`` of type ``double`` and ``b`` of type ``double``

Since ``+`` is defined for ``units::Unit`` the same lambda can be used for data and unit.
This call to ``transform`` will add the two variables (or variable views) ``a`` and ``b`` and return a new variable.

.. code-block:: cpp

    auto var = transform<
        std::tuple<std::tuple<double, float>, std::tuple<double, double>>>(
        a, b, [](const auto &a_, const auto &b_) { return a_ + b_; });

Example 2 
~~~~~~~~~

In-place transform with two variables and special unit handling, using the help ``overloaded``, to define an "overloaded" lambda:
This call to ``transform_in_place``  accepts two variables (or variable views) ``a`` and ``b``.
``a`` is modified in-place, no new variable is returned.

.. code-block:: cpp

    transform_in_place<std::tuple<bool>>(
        a, b,
        overloaded{[](auto &a_, const auto &b_) { a &= b; },
                   [](const units::Unit &a, const units::Unit &b) {
                     if (a != b)
                       throw std::runtime_error("Unit must match");
                   }});

``transform_in_place`` modifies its first argument.
Note that the template argument ``std::tuple<bool>`` is equivalent to ``std::tuple<std::tuple<bool, bool>>``, i.e., both arguments are require have element type ``bool``.

Recommended usage
-----------------

For improved testability and maintainability it is recommended to define the operator stand-alone instead of inline.
The list of supported types can also be provided in this manner.

- Use ``arg_list`` to define list of supported type combinations.
- Flags in ``transform_flags.h`` can be used to, e.g., prevent code generation if an argument has variances.

Example
~~~~~~~

If operation is added to ``namespace scipp::variable``, define:

.. code-block:: cpp

   // In scipp/core/include/scipp/core/element/my_op.h:
   namespace scipp::core::element {
   constexpr auto my_op = overloaded{
       arg_list<std::tuple<double, int64_t>, std::tuple<double, int32_t>>,
       transform_flags::expect_no_variance_arg<0>,
       [](const auto &a, const auto &b) { return a + b; }};
   };

.. code-block:: cpp

   // In scipp/core/include/scipp/variable/my_op.h:
   namespace scipp::variable {
   Variable my_op(const VariableConstView &a, const VariableConstView &b) {
       return transform(a, b, core::element::my_op);
   }

- Here, variances for the first argument are disabled explicitly.
- Unit tests should be written independently for ``scipp::core::element::my_op``.
- ``scipp::variable::my_op`` should only have essential tests relying on correctness of ``transform`` and ``scipp::core::element::my_op``.
