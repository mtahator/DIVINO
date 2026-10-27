"""Figure F7: representative forward field comparisons FNO vs body-fitted FEM vs |error|,
one row per physics regime (insulated, source, Robin, rotated, two-cavity)."""
import argparse
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import divino as L
import os
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)  # output folders (relative to src/)
from fem import solve_fem, fem_to_grid

REGIMES = [
    ('insulated', {'cavities': [(0.50, 0.50, 0.18, 4.0, 0.0)], 'Q0': 0.0, 'h': 0.0}),
    ('source Q0=2.5', {'cavities': [(0.50, 0.50, 0.18, 3.0, 0.0)], 'Q0': 2.5, 'h': 0.0}),
    ('Robin h=6', {'cavities': [(0.50, 0.50, 0.18, 5.0, 0.0)], 'Q0': 0.0, 'h': 6.0}),
    ('rotated+source+Robin', {'cavities': [(0.45, 0.55, 0.17, 6.0, 0.6)], 'Q0': 1.5, 'h': 3.0}),
    ('two cavities', {'cavities': [(0.34, 0.40, 0.12, 3.0, 0.0), (0.66, 0.60, 0.12, 8.0, 0.5)], 'Q0': 0.0, 'h': 2.0}),
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='checkpoints/pino.pt'); a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, _ = L.load_model(a.ckpt, dev); N = cfg['N']
    fig, ax = plt.subplots(len(REGIMES), 3, figsize=(12, 4 * len(REGIMES)))
    for row, (name, case) in enumerate(REGIMES):
        T, xs = L.fno_eval(model, case, N, dev)
        m, th = solve_fem(case, hf=0.011); Tf, Xf, Yf = fem_to_grid(m, th, N)
        dx = 1.0 / (N - 1); margin = np.zeros((N, N), bool)
        for c in case['cavities']: margin |= L.cav_phi_np(Xf, Yf, c) < 2.5 * dx
        mk = (~np.isnan(T)) & (~np.isnan(Tf)) & (~margin)
        rel = np.sqrt(np.mean((T - Tf)[mk] ** 2)) / (np.sqrt(np.mean(Tf[mk] ** 2)) + 1e-9)
        err = np.abs(T - Tf); vmin, vmax = np.nanmin(Tf), np.nanmax(Tf)
        for k, (U, ti, cm) in enumerate([(T, 'FNO', 'inferno'), (Tf, 'FEM', 'inferno'), (err, '|err|', 'hot_r')]):
            kw = dict(cmap=cm, origin='lower', extent=[0, 1, 0, 1])
            if k < 2: kw.update(vmin=vmin, vmax=vmax)
            im = ax[row, k].imshow(U, **kw)
            for c in case['cavities']:
                o = L.cav_outline(c); ax[row, k].plot(o[:, 0], o[:, 1], 'c' if k < 2 else 'b', lw=1.2)
            ax[row, k].set_xticks([]); ax[row, k].set_yticks([])
            plt.colorbar(im, ax=ax[row, k], fraction=0.046)
        ax[row, 0].set_ylabel(f'{name}\nrelL2={rel*100:.2f}%', fontsize=10)
    plt.tight_layout(); plt.savefig('figures/F7_field_comparison.png', dpi=120); print('saved figures/F7_field_comparison.png')

if __name__ == '__main__':
    main()
