"""
Global RHS assembler.

Supports 1D and 2D, with optional prescribed electromagnetic backgrounds.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.geometry.base_metric import BaseMetric
from axion_em_gr.physics.axion import compute_axion_rhs
from axion_em_gr.physics.background_em import BackgroundEMField
from axion_em_gr.physics.maxwell import compute_maxwell_rhs
from axion_em_gr.physics.potentials import AxionPotential
from axion_em_gr.physics.sources import SourceModel, VacuumSources


class RHSComputer:
    """
    Callable RHS object.

    This class stores the fixed ingredients needed to compute du/dt.

    If background_em is provided, the state used by the RHS is first updated
    with the prescribed E^i(t,x) and B^i(t,x). This is useful for reduced
    scenarios where the EM field is imposed externally, e.g. a rotating dipole.
    """

    def __init__(
        self,
        grid: Grid,
        metric: BaseMetric,
        potential: AxionPotential,
        numerics: NumericalParameters,
        physical: PhysicalParameters | None = None,
        sources: SourceModel | None = None,
        evolve_axion: bool = True,
        evolve_maxwell: bool = False,
        include_axion_em_coupling: bool = False,
        background_em: BackgroundEMField | None = None,
        background_em_mode: str = "replace",
    ) -> None:
        if background_em_mode not in ("replace", "add"):
            raise ValueError("background_em_mode must be 'replace' or 'add'.")

        self.grid = grid
        self.metric = metric
        self.potential = potential
        self.numerics = numerics
        self.physical = PhysicalParameters() if physical is None else physical
        self.sources = VacuumSources() if sources is None else sources
        self.evolve_axion = evolve_axion
        self.evolve_maxwell = evolve_maxwell
        self.include_axion_em_coupling = include_axion_em_coupling
        self.background_em = background_em
        self.background_em_mode = background_em_mode

    def state_with_background(self, state: State, t: float) -> State:
        """
        Return a copy of state with prescribed EM background applied.

        If background_em_mode == "replace":
            E and B are set equal to the background.

        If background_em_mode == "add":
            background is added to the existing E and B.
        """
        if self.background_em is None:
            return state

        E_bg, B_bg = self.background_em.evaluate(t, self.grid)

        work = state.copy()

        if self.background_em_mode == "replace":
            work.E = E_bg
            work.B = B_bg
            return work

        # add mode
        if work.E is None:
            work.E = E_bg
        else:
            work.E = work.E + E_bg

        if work.B is None:
            work.B = B_bg
        else:
            work.B = work.B + B_bg

        return work

    def __call__(self, state: State, t: float) -> State:
        state_eval = self.state_with_background(state, t)
        geom = self.metric.evaluate(t, self.grid)

        if self.evolve_axion:
            rhs_a, rhs_Pi = compute_axion_rhs(
                state=state_eval,
                grid=self.grid,
                geom=geom,
                potential=self.potential,
                numerics=self.numerics,
                physical=self.physical,
                include_em_coupling=self.include_axion_em_coupling,
            )
        else:
            rhs_a = np.zeros_like(state.a)
            rhs_Pi = np.zeros_like(state.Pi)

        rhs_E = None
        rhs_B = None

        if self.evolve_maxwell:
            rhs_E, rhs_B = compute_maxwell_rhs(
                state=state_eval,
                t=t,
                grid=self.grid,
                geom=geom,
                sources=self.sources,
                numerics=self.numerics,
                physical=self.physical,
                include_axion_coupling=self.include_axion_em_coupling,
            )
        else:
            if state.E is not None:
                rhs_E = np.zeros_like(state.E)
            elif state_eval.E is not None:
                rhs_E = np.zeros_like(state_eval.E)

            if state.B is not None:
                rhs_B = np.zeros_like(state.B)
            elif state_eval.B is not None:
                rhs_B = np.zeros_like(state_eval.B)

        return State(
            a=rhs_a,
            Pi=rhs_Pi,
            E=rhs_E,
            B=rhs_B,
        )