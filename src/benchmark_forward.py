"""WP-C: statistical forward benchmark of the PINO vs body-fitted FEM.
In-distribution + OOD test sets; saves results/forward_metrics.csv, Table T1 (printed),
and Figure F1 (error histogram + error-vs-(r,a) heat map)."""
import os, time, argparse
import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import divino as L
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)  # output folders (relative to src/)
from fem import solve_fem, fem_to_grid

def metrics(Tf, case, N, fem_hf=0.011):
    try:
        m, th = solve_fem(case, hf=fem_hf)
        Tref, X, Y = fem_to_grid(m, th, N)
    except Exception:
        return np.nan, np.nan, np.nan
    dx = 1.0 / (N - 1); margin = np.zeros((N, N), bool)
    for c in case['cavities']:
        margin |= L.cav_phi_np(X, Y, c) < 2.5 * dx
    mk = (~np.isnan(Tf)) & (~np.isnan(Tref)) & (~margin)
    if mk.sum() < 100:
        return np.nan, np.nan, np.nan
    d = np.abs(Tf - Tref)[mk]; denom = np.sqrt(np.mean(Tref[mk] ** 2)) + 1e-9
    rel = np.sqrt(np.mean(d ** 2)) / denom
    en = abs(np.nanmean((Tf - Tref)[mk]))
    return rel, float(d.max()), en

def run_set(model, rng, n, N, dev, ood):
    rows = []
    for i in range(n):
        case = L.sample_case(rng, two_cav_prob=0.5, ood=ood)
        Tf, _ = L.fno_eval(model, case, N, dev)
        rel, linf, en = metrics(Tf, case, N)
        c0 = case['cavities'][0]
        rows.append(dict(idx=i, ncav=len(case['cavities']), r=c0[2], a=c0[3], cx=c0[0], cy=c0[1],
                         phi=c0[4], Q0=case['Q0'], h=case['h'], relL2=rel, Linf=linf, bias=en,
                         split='OOD' if ood > 0 else 'in-dist'))
    return rows

def stats(df):
    return dict(mean=df.relL2.mean(), median=df.relL2.median(), p95=df.relL2.quantile(0.95),
                max=df.relL2.max(), std=df.relL2.std())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/pino.pt')
    ap.add_argument('--n_indist', type=int, default=150)
    ap.add_argument('--n_ood', type=int, default=60)
    ap.add_argument('--ood', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=123)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, _ = L.load_model(a.ckpt, dev); N = cfg['N']
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    rows = run_set(model, rng, a.n_indist, N, dev, ood=0.0)
    rows += run_set(model, rng, a.n_ood, N, dev, ood=a.ood)
    df = pd.DataFrame(rows); df.to_csv('results/forward_metrics.csv', index=False)
    print(f'benchmark: {len(df)} cases in {time.time()-t0:.0f}s -> results/forward_metrics.csv')
    print('\n=== Table T1: forward relL2 statistics ===')
    print(f'{"split":>10} | {"N":>4} | {"mean":>9} | {"median":>9} | {"p95":>9} | {"max":>9}')
    for sp in ['in-dist', 'OOD']:
        s = stats(df[df.split == sp]); n = (df.split == sp).sum()
        print(f'{sp:>10} | {n:>4} | {s["mean"]:9.2e} | {s["median"]:9.2e} | {s["p95"]:9.2e} | {s["max"]:9.2e}')
    # Figure F1
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for sp, c in [('in-dist', 'C0'), ('OOD', 'C3')]:
        ax[0].hist(df[df.split == sp].relL2 * 100, bins=30, alpha=0.55, label=sp, color=c)
    ax[0].set_xlabel('relL2 [%]'); ax[0].set_title('Forward error distribution'); ax[0].legend()
    di = df[df.split == 'in-dist']
    sc = ax[1].scatter(di.r, di.a, c=di.relL2 * 100, cmap='viridis', s=24)
    ax[1].set_xlabel('r'); ax[1].set_ylabel('a'); ax[1].set_title('relL2 [%] vs (r,a)'); plt.colorbar(sc, ax=ax[1])
    sc2 = ax[2].scatter(di.h, di.Q0, c=di.relL2 * 100, cmap='magma', s=24)
    ax[2].set_xlabel('h (Robin)'); ax[2].set_ylabel('Q0 (source)'); ax[2].set_title('relL2 [%] vs (h,Q0)'); plt.colorbar(sc2, ax=ax[2])
    plt.tight_layout(); plt.savefig('figures/F1_forward_error.png', dpi=130); print('saved figures/F1_forward_error.png')

if __name__ == '__main__':
    main()
