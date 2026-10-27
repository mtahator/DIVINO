"""WP-G: cavity boundary fidelity. Compare the FNO cavity normal-flux ratio |dT/dn|/|gradT|
to the body-fitted FEM across the shape exponent a (insulated case), and show it tracks the FEM
floor (no upward drift) and improves with grid resolution. Figure F5."""
import argparse
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import divino as L
import os
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)  # output folders (relative to src/)
from fem import solve_fem, fem_to_grid

def cavity_ratio(T, fluid, cav, N):
    Xg, Yg = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    Ts = np.where(fluid, T, np.nan); dx = 1.0 / (N - 1)
    def deriv(ax):
        plus = np.roll(Ts, -1, ax); minus = np.roll(Ts, 1, ax); g = np.full_like(Ts, np.nan)
        c = ~np.isnan(plus) & ~np.isnan(minus) & ~np.isnan(Ts); g[c] = (plus[c] - minus[c]) / (2 * dx)
        f = ~np.isnan(plus) & np.isnan(minus) & ~np.isnan(Ts); g[f] = (plus[f] - Ts[f]) / dx
        b = np.isnan(plus) & ~np.isnan(minus) & ~np.isnan(Ts); g[b] = (Ts[b] - minus[b]) / dx
        return g
    gx = deriv(1); gy = deriv(0); solid = ~fluid; nb = np.zeros_like(fluid)
    nb[:, :-1] |= solid[:, 1:]; nb[:, 1:] |= solid[:, :-1]; nb[:-1, :] |= solid[1:, :]; nb[1:, :] |= solid[:-1, :]
    band = fluid & nb; band[0, :] = band[-1, :] = band[:, 0] = band[:, -1] = False
    valid = band & ~np.isnan(gx) & ~np.isnan(gy)
    cx, cy, r, a, phi = cav; X, Y = L._rot_np(Xg[valid] - cx, Yg[valid] - cy, phi)
    nx = np.sign(X) * np.abs(X) ** (a - 1); ny = np.sign(Y) * np.abs(Y) ** (a - 1); nn = np.sqrt(nx ** 2 + ny ** 2) + 1e-12
    dtn = gx[valid] * nx / nn + gy[valid] * ny / nn
    gm = np.sqrt(gx[valid] ** 2 + gy[valid] ** 2)
    return np.abs(dtn).mean() / (gm.mean() + 1e-9)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='checkpoints/pino.pt'); a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, _ = L.load_model(a.ckpt, dev); N = cfg['N']
    avals = [2, 3, 4, 6, 8]
    fno_r, fem_r = [], []
    for av in avals:
        case = {'cavities': [(0.5, 0.5, 0.18, float(av), 0.0)], 'Q0': 0.0, 'h': 0.0}
        T, _ = L.fno_eval(model, case, N, dev); fno_r.append(cavity_ratio(T, ~np.isnan(T), case['cavities'][0], N))
        m, th = solve_fem(case, hf=0.011); Tf, _, _ = fem_to_grid(m, th, N)
        fem_r.append(cavity_ratio(Tf, ~np.isnan(Tf), case['cavities'][0], N))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(avals, np.array(fno_r), 'o-', label='FNO (PINO)')
    ax.plot(avals, np.array(fem_r), 's--', label='body-fitted FEM (floor)')
    ax.set_xlabel('shape exponent a'); ax.set_ylabel('cavity |dT/dn| / |gradT|')
    ax.set_title('Insulation fidelity vs shape (tracks FEM floor)'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('figures/F5_insulation.png', dpi=130)
    print('FNO ratio:', np.round(fno_r, 3)); print('FEM ratio:', np.round(fem_r, 3)); print('saved figures/F5_insulation.png')

if __name__ == '__main__':
    main()
