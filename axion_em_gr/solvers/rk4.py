"""
Explicit fourth-order Runge-Kutta integrator.
"""

from __future__ import annotations

from collections.abc import Callable

from axion_em_gr.core.state import State


class RK4:
    """
    Classical explicit RK4 integrator.
    """

    def step(
        self,
        state: State,
        t: float,
        dt: float,
        rhs_func: Callable[[State, float], State],
    ) -> State:
        k1 = rhs_func(state, t)

        s2 = state.add_scaled(k1, 0.5 * dt)
        k2 = rhs_func(s2, t + 0.5 * dt)

        s3 = state.add_scaled(k2, 0.5 * dt)
        k3 = rhs_func(s3, t + 0.5 * dt)

        s4 = state.add_scaled(k3, dt)
        k4 = rhs_func(s4, t + dt)

        new_state = State.linear_combination(
            base=state,
            terms=[
                (dt / 6.0, k1),
                (dt / 3.0, k2),
                (dt / 3.0, k3),
                (dt / 6.0, k4),
            ],
        )

        return new_state
