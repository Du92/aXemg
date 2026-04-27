"""
Evolution driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from axion_em_gr.core.boundary import BoundaryCondition
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import contract_cov_contra, lower_vector
from axion_em_gr.physics.constraints import (
    constraint_norms,
    electric_constraint,
    magnetic_constraint,
)
from axion_em_gr.solvers.rk4 import RK4


@dataclass
class EvolutionHistory:
    """
    Lightweight container for diagnostic history.
    """

    times: list[float] = field(default_factory=list)

    max_a: list[float] = field(default_factory=list)
    l2_a: list[float] = field(default_factory=list)
    max_Pi: list[float] = field(default_factory=list)

    max_E: list[float] = field(default_factory=list)
    max_B: list[float] = field(default_factory=list)
    l2_em: list[float] = field(default_factory=list)

    max_EdotB: list[float] = field(default_factory=list)
    l2_EdotB: list[float] = field(default_factory=list)

    l2_div_B: list[float] = field(default_factory=list)
    linf_div_B: list[float] = field(default_factory=list)
    l2_div_E: list[float] = field(default_factory=list)
    linf_div_E: list[float] = field(default_factory=list)

    snapshot_times: list[float] = field(default_factory=list)
    snapshots: list[State] = field(default_factory=list)

class EvolutionSolver:
    """
    Time-evolution driver.
    """

    def __init__(
        self,
        grid: Grid,
        rhs_computer: RHSComputer,
        integrator: RK4,
        boundary: BoundaryCondition,
        numerics: NumericalParameters,
        save_snapshots: bool = False,
        snapshot_every: int | None = None,) -> None:
        self.grid = grid
        self.rhs_computer = rhs_computer
        self.integrator = integrator
        self.boundary = boundary
        self.numerics = numerics
        self.save_snapshots = save_snapshots
        self.snapshot_every = snapshot_every

    def _rhs_with_boundary(self, state: State, t: float) -> State:
        """
        Apply boundary conditions before computing the RHS.

        This is important for intermediate RK4 stages.
        """
        work = state.copy()
        self.boundary.apply_state(work, self.grid)
        return self.rhs_computer(work, t)

    def _record_diagnostics(
        self,
        state: State,
        t: float,
        history: EvolutionHistory,
    ) -> None:
        state = self.rhs_computer.state_with_background(state, t)

        interior = self.grid.interior_slices
        if self.grid.ndim == 1:
            volume_element = self.grid.dx[0]
        elif self.grid.ndim == 2:
            volume_element = self.grid.dx[0] * self.grid.dx[1]
        else:
            raise NotImplementedError("Diagnostics currently support 1D and 2D.")

        history.times.append(t)

        a_int = state.a[interior]
        Pi_int = state.Pi[interior]

        history.max_a.append(float(np.max(np.abs(a_int))))
        history.max_Pi.append(float(np.max(np.abs(Pi_int))))
        history.l2_a.append(float(np.sqrt(np.sum(a_int**2) * volume_element)))

        if state.E is not None and state.B is not None:
            E_int = state.E[(slice(None), *interior)]
            B_int = state.B[(slice(None), *interior)]

            E2 = np.sum(E_int**2, axis=0)
            B2 = np.sum(B_int**2, axis=0)

            history.max_E.append(float(np.max(np.sqrt(E2))))
            history.max_B.append(float(np.max(np.sqrt(B2))))
            history.l2_em.append(float(np.sqrt(np.sum(E2 + B2) * volume_element)))

            geom = self.rhs_computer.metric.evaluate(t, self.grid)
            E_down = lower_vector(state.E, geom.gamma_down)
            EdotB = contract_cov_contra(E_down, state.B)
            EdotB_int = EdotB[interior]

            history.max_EdotB.append(float(np.max(np.abs(EdotB_int))))
            history.l2_em.append(float(np.sqrt(np.sum(E2 + B2) * volume_element)))

            sources = self.rhs_computer.sources

            div_B = magnetic_constraint(
                state=state,
                grid=self.grid,
                geom=geom,
                numerics=self.numerics,)

            div_E = electric_constraint(
                state=state,
                t=t,
                grid=self.grid,
                geom=geom,
                sources=sources,
                numerics=self.numerics,
                physical=self.rhs_computer.physical,
                include_axion_coupling=self.rhs_computer.include_axion_em_coupling,)

            l2_B, linf_B = constraint_norms(div_B, self.grid)
            l2_E, linf_E = constraint_norms(div_E, self.grid)

            history.l2_div_B.append(l2_B)
            history.linf_div_B.append(linf_B)
            history.l2_div_E.append(l2_E)
            history.linf_div_E.append(linf_E)

    def evolve(self, state0: State) -> tuple[State, EvolutionHistory]:
        """
        Evolve the initial state.

        Returns
        -------
        final_state, history
        """
        state = state0.copy()
        self.boundary.apply_state(state, self.grid)

        t = 0.0
        dt = self.numerics.dt
        t_final = self.numerics.t_final

        history = EvolutionHistory()
        self._record_diagnostics(state, t, history)
        self._record_snapshot(state, t, step=0, history=history)

        step = 0

        while t < t_final:
            if t + dt > t_final:
                dt = t_final - t

            state = self.integrator.step(
                state=state,
                t=t,
                dt=dt,
                rhs_func=self._rhs_with_boundary,
            )

            self.boundary.apply_state(state, self.grid)
            state.assert_finite()

            t += dt
            step += 1

            if step % self.numerics.output_every == 0:
                self._record_diagnostics(state, t, history)
                self._record_snapshot(state, t, step=step, history=history)

                message = (
                    f"step={step:06d}  "
                    f"t={t:.6f}  "
                    f"max|a|={history.max_a[-1]:.6e}  "
                    f"L2(a)={history.l2_a[-1]:.6e}"
                )

                if state.E is not None and state.B is not None:
                    message += (
                        f"  max|E|={history.max_E[-1]:.6e}"
                        f"  max|B|={history.max_B[-1]:.6e}"
                        f"  max|E.B|={history.max_EdotB[-1]:.3e}"
                        f"  L2(divB)={history.l2_div_B[-1]:.3e}"
                        f"  L2(divE)={history.l2_div_E[-1]:.3e}"
                    )

                print(message)

        self._record_diagnostics(state, t, history)
        if self.save_snapshots:
            history.snapshot_times.append(t)
            history.snapshots.append(state.copy())

        return state, history
    
    def _record_snapshot(
            self,
            state: State,
            t: float,
            step: int,
            history: EvolutionHistory,
            ) -> None:
            """
            Store a full copy of the state at selected times.
            This is useful for later analysis and visualization, but can consume a lot of memory if used too frequently.
            """
            if not self.save_snapshots:
                return

            if self.snapshot_every is None:
                return

            if step % self.snapshot_every != 0:
                return

            state_to_store = self.rhs_computer.state_with_background(state, t)

            history.snapshot_times.append(t)
            history.snapshots.append(state_to_store.copy())