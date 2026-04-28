# Limitations and future development

This document lists the current limitations of `axion_em_gr`.

---

## 1. Fixed background approximation

The code does not evolve the Einstein equations.

The metric is prescribed through fixed background models such as:

- flat spacetime,
- TT gravitational-wave background,
- isotropic Schwarzschild-inspired metric,
- smooth compact-object toy metric.

Therefore, the simulations do not include backreaction of the axion or electromagnetic fields on spacetime.

---

## 2. Reduced dimensionality

The current implementation supports 1D and 2D reduced domains.

The electromagnetic fields still keep three vector components, but the spatial dependence is reduced.

This is useful for controlled studies, but it is not equivalent to a full 3D compact-object simulation.

---

## 3. Prescribed rotating dipole

The rotating dipole module is a prescribed electromagnetic background.

It is not:

- a full Deutsch solution,
- a force-free magnetosphere,
- a plasma simulation,
- a resistive magnetosphere,
- a self-consistent Maxwell solution with stellar boundary conditions.

It is intended for exploratory studies of axion response to a time-dependent magnetized compact-object environment.

---

## 4. Plasma physics

The current code does not include:

- plasma frequency,
- force-free currents,
- resistive currents,
- pair creation,
- magnetospheric charge starvation,
- Goldreich-Julian density evolution.

These can be added later through source models and modified Maxwell closures.

---

## 5. Boundary conditions

Boundary conditions are modular but not yet characteristic.

Available options include:

- periodic,
- outflow,
- Dirichlet,
- Neumann,
- Sommerfeld-like,
- mixed field-wise boundaries.

For high-accuracy compact-object wave extraction, more careful characteristic or absorbing boundaries would be needed.

---

## 6. Constraint cleaning

The electric constraint cleaning supports flat and curved Poisson solvers.

However, the cleaning is not yet fully mimetic. The discrete Poisson operator, gradient correction and divergence diagnostic are not guaranteed to form an exactly compatible discrete complex.

Therefore, the cleaning is expected to reduce the constraint norm, but not necessarily eliminate it to machine precision.

---

## 7. Numerical methods

The current default scheme is finite differences plus explicit RK4.

The code does not yet include:

- high-order finite-volume methods,
- discontinuous Galerkin methods,
- spectral methods,
- AMR,
- implicit time stepping,
- symplectic integrators,
- GPU acceleration.

---

## 8. Physical units

Many examples use dimensionless units.

Care is needed before interpreting amplitudes, luminosities or frequencies in physical units.

The current examples are designed primarily to isolate mechanisms and compare relative responses.

---

## 9. Future directions

Possible future extensions include:

- full 3D grid support,
- improved mimetic operators,
- characteristic boundary conditions,
- plasma and force-free closures,
- physically calibrated neutron-star magnetospheres,
- binary neutron-star near-zone fields,
- improved GW background modules,
- adaptive time stepping,
- high-order convergence tests,
- packaging and DOI release.
