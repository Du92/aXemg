# Physical formulation

This document summarizes the physical equations implemented in `axion_em_gr`.

---

## 1. Conventions

The code uses the metric signature

$$(-,+,+,+).$$

The spacetime is foliated by spatial hypersurfaces $$\Sigma_t$$, with line element

$$ds^2 = -N^2dt^2 + \gamma_{ij} (dx^i+\beta^idt) (dx^j+\beta^jdt).$$

Here:

- $$N$$ is the lapse,
- $$\beta^i$$ is the shift,
- $$\gamma_{ij}$$ is the spatial metric,
- $$\gamma^{ij}$$ is the inverse spatial metric,
- $$\gamma=\det(\gamma_{ij})$$,
- $$K$$ is the trace of the extrinsic curvature.

The spatial covariant derivative associated with $$\gamma_{ij}$$is denoted by$$D_i$$.

The dual Faraday tensor is defined as

$$\tilde F^{\mu\nu} = \frac{1}{2} \varepsilon^{\mu\nu\rho\sigma} F_{\rho\sigma}.$$

The electric and magnetic fields are those measured by Eulerian observers.

---

## 2. Axion electrodynamics

The implemented system is based on the axion-electrodynamics action

$$S = \int d^4x\sqrt{-g} \left[ -\frac{1}{4}F_{\mu\nu}F^{\mu\nu} -\frac{1}{2}\nabla_\mu a\nabla^\mu a - V(a) - \frac{g_{a\gamma}}{4}aF_{\mu\nu}\tilde F^{\mu\nu} \right].$$

The default potential is

$$V(a) = \frac{1}{2}m_a^2a^2.$$

The code allows other potentials by implementing the interface

$$\frac{dV}{da}.$$

---

## 3. Evolved axion variables

The evolved scalar variables are

$$a(t,x^i), \qquad \Pi(t,x^i).$$

The 3+1 evolution system is

$$\partial_t a = N\Pi + \beta^i\partial_i a,$$

$$\partial_t \Pi = \beta^i\partial_i\Pi + N \left[ D_iD^i a + K\Pi - D_i(\ln N)D^i a - \frac{dV}{da} - g_{a\gamma}E_iB^i \right].$$

The scalar Laplacian is implemented as

$$D_iD^ia = \frac{1}{\sqrt{\gamma}} \partial_i \left( \sqrt{\gamma}\gamma^{ij}\partial_j a \right).$$

---

## 4. Electromagnetic variables

The evolved electromagnetic variables are

$$E^i(t,x^j), \qquad B^i(t,x^j).$$

The code keeps track of the distinction between contravariant fields $$E^i,B^i$$ and covariant fields

$$E_i=\gamma_{ij}E^j, \qquad B_i=\gamma_{ij}B^j.$$

The axion source term uses

$$E_iB^i = \gamma_{ij}E^jB^i.$$

---

## 5. Maxwell constraints

The magnetic constraint is

$$D_iB^i=0.$$

The axion-modified electric constraint is

$$D_iE^i = \rho - g_{a\gamma}B^iD_i a.$$

The code monitors the residual

$$C_E = D_iE^i - \rho + g_{a\gamma}B^iD_i a.$$

A constrained state satisfies

$$C_E=0.$$

---

## 6. Maxwell evolution

The Maxwell evolution equations are

$$\partial_t B^i = \mathcal{L}_\beta B^i + NK B^i - \epsilon^{ijk}D_j(NE_k),$$

$$\partial_t E^i = \mathcal{L}_\beta E^i + NK E^i - \epsilon^{ijk}D_j(NB_k) + Nj^i + g_{a\gamma}N\epsilon^{ijk}E_kD_ja - g_{a\gamma}N\Pi B^i.$$

The Lie derivative of a contravariant vector is

$$(\mathcal{L}_\beta V)^i = \beta^j\partial_jV^i - V^j\partial_j\beta^i.$$

The spatial Levi-Civita tensor is represented as

$$\epsilon^{ijk} = \frac{[ijk]}{\sqrt{\gamma}},$$

where $$[ijk]$$ is the antisymmetric Levi-Civita symbol.

---

## 7. Curved two-dimensional operators

For 2D fixed-background simulations, the scalar Laplacian is

$$D_iD^i f = \frac{1}{\sqrt{\gamma}} \partial_i \left( \sqrt{\gamma}\gamma^{ij}\partial_j f \right).$$

The covariant divergence is

$$D_iV^i = \frac{1}{\sqrt{\gamma}} \partial_i \left( \sqrt{\gamma}V^i \right).$$

For a covector $$A_i$$, the curl operator uses

$$\epsilon^{ijk}D_jA_k = \epsilon^{ijk}\partial_jA_k.$$

The Christoffel contribution cancels because it is symmetric in the lower indices and contracted with the antisymmetric Levi-Civita tensor.

---

## 8. Electric constraint cleaning

The flat electric constraint cleaner solves

$$\nabla^2\phi=C_E,$$

and corrects

$$E^i\rightarrow E^i-\partial^i\phi.$$

The curved cleaner solves

$$D_iD^i\phi=C_E,$$

and corrects

$$E^i\rightarrow E^i-D^i\phi = E^i-\gamma^{ij}\partial_j\phi.$$

In discrete form, the cleaning may reduce the global norm of the constraint without making the pointwise $$L_\infty$$ norm vanish exactly. This is expected because the Poisson operator, correction operator and diagnostic divergence are not yet fully mimetic.

---

## 9. Implemented metrics

### Flat metric

$$N=1, \qquad \beta^i=0, \qquad \gamma_{ij}=\delta_{ij}, \qquad K=0.$$

### Isotropic Schwarzschild metric

$$ds^2 = -\alpha^2dt^2 + \psi^4 (dx^2+dy^2+dz^2),$$

with

$$\psi = 1+\frac{M}{2r}, \qquad \alpha = \frac{1-\frac{M}{2r}}{1+\frac{M}{2r}}.$$

Thus

$$\gamma_{ij}=\psi^4\delta_{ij}, \qquad \gamma^{ij}=\psi^{-4}\delta^{ij}, \qquad \sqrt{\gamma}=\psi^6.$$

### Smooth compact-object toy metric

The smooth compact-object metric is not an exact solution of Einstein's equations. It is a controlled fixed background intended for reduced compact-object experiments.

It uses a conformally flat spatial metric and a smooth lapse profile.

---

## 10. Prescribed rotating dipole

A rotating dipole background can be prescribed as

$$\boldsymbol{\mu}(t) = \mu_0 \left( \sin\chi\cos(\Omega t+\phi_0), \sin\chi\sin(\Omega t+\phi_0), \cos\chi \right).$$

The magnetic field is approximated by the flat dipole expression

$$\mathbf{B} = B_{\rm scale} \frac{ 3(\boldsymbol{\mu}\cdot\hat{\mathbf r})\hat{\mathbf r} - \boldsymbol{\mu} }{r_{\rm eff}^3}.$$

An approximate induced electric field may be included through

$$\mathbf{E} = -\mathbf{v}_{\rm rot}\times\mathbf{B}.$$

A small parallel electric component can be added phenomenologically:

$$\mathbf{E}_\parallel = \epsilon_\parallel f(r)\mathbf{B}.$$

This is a prescribed electromagnetic background, not a self-consistent plasma magnetosphere.

---

## 11. Limitations

The current physical implementation assumes:

- fixed spacetime backgrounds,
- no evolution of the Einstein equations,
- no self-consistent plasma,
- no force-free constraints,
- no resistive terms,
- no adaptive mesh refinement,
- reduced 1D or 2D domains.

The code is designed for controlled numerical experiments and exploratory model building.
