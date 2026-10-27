"""Validate the grid Deep-Ritz discretization (source + Robin + multi-cavity) against the
body-fitted FEM. If they agree to the staircase floor, the energy/physics is correct."""
import numpy as np
from fem import solve_fem, fem_to_grid
import divino

N = 160
cases = {
    'insulated 1-cav':     {'cavities': [(0.50, 0.50, 0.20, 4.0, 0.0)], 'Q0': 0.0, 'h': 0.0},
    'Robin h=5':           {'cavities': [(0.50, 0.50, 0.20, 4.0, 0.0)], 'Q0': 0.0, 'h': 5.0},
    'source Q0=2':         {'cavities': [(0.50, 0.50, 0.20, 4.0, 0.0)], 'Q0': 2.0, 'h': 0.0},
    'rotated+Q+Robin':     {'cavities': [(0.45, 0.55, 0.18, 6.0, 0.6)], 'Q0': 1.0, 'h': 2.0},
    'two cavities':        {'cavities': [(0.35, 0.40, 0.13, 3.0, 0.0), (0.65, 0.62, 0.12, 8.0, 0.5)], 'Q0': 0.0, 'h': 0.0},
    'two+source+Robin':    {'cavities': [(0.36, 0.42, 0.12, 3.0, 0.2), (0.66, 0.60, 0.12, 7.0, 0.4)], 'Q0': 1.5, 'h': 3.0},
}
print(f'{"case":>20} | {"relL2(grid vs FEM)":>18} | {"Linf":>8}')
for name, case in cases.items():
    Tg, xs = divino.direct_grid_solve(case, N)
    m, th = solve_fem(case, hf=0.010)
    Tf, X, Y = fem_to_grid(m, th, N)
    # mask cavities + a 2dx margin band where staircase/boundary interpolation is noisy
    dx = 1.0 / (N - 1)
    margin = np.zeros((N, N), bool)
    for c in case['cavities']:
        margin |= divino.cav_phi_np(X, Y, c) < 2.5 * dx
    mask = (~np.isnan(Tg)) & (~np.isnan(Tf)) & (~margin)
    d = np.abs(Tg - Tf)[mask]
    rel = np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(Tf[mask] ** 2))
    print(f'{name:>20} | {rel:18.3e} | {d.max():8.3e}')
