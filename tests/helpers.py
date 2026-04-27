"""
Small helper functions for tests.
"""


def core_slices(grid, margin: int = 4):
    """
    Return slices that exclude both ghost zones and a few physical cells near
    the boundary.

    This is useful for tests of composed finite-difference operators, where
    the largest errors are often located next to the boundary/ghost interface.
    """
    g = grid.nghost

    if grid.ndim == 1:
        n = grid.shape[0]
        return slice(g + margin, g + n - margin)

    if grid.ndim == 2:
        nx, ny = grid.shape
        return (
            slice(g + margin, g + nx - margin),
            slice(g + margin, g + ny - margin),
        )

    raise NotImplementedError("Only 1D and 2D grids are supported in tests.")