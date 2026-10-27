"""WP-F: convergence / ablation / multi-seed (reduced-scale). Trains short PINOs over a small
hyperparameter sweep + seeds, evaluates relL2 vs body-fitted FEM on a fixed test set.
Outputs results/ablation.csv (Table T5) and figures/F4_ablation.png."""
import argparse, time
import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import divino as L
import os
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)  # output folders (relative to src/)
from fem import solve_fem, fem_to_grid

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fixed_testset(n, N, seed=999):
    rng = np.random.default_rng(seed); cases = [L.sample_case(rng, 0.4) for _ in range(n)]
    refs = []
    for c in cases:
        m, th = solve_fem(c, hf=0.012); Tg, X, Y = fem_to_grid(m, th, N); refs.append((Tg, X, Y))
    return cases, refs

def train_short(N, modes, width, layers, steps, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X, Y, xs = L.make_grid(N, dev); Dm, Dv = L.dirichlet_fields(N, dev); dx = 1 / (N - 1)
    m = L.FNO2d(6, width, modes, layers).to(dev); opt = torch.optim.Adam(m.parameters(), 2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=2e-4)
    for it in range(steps):
        P, _ = L.sample_params(rng, 12, dev, 0.4); inp, M, _ = L.channels_from_params(P, X, Y)
        loss = L.ritz_energy(L.hard_bc(m(inp), Dm, Dv), M, P, dx)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m, Dm, Dv

def evaluate(m, Dm, Dv, cases, refs, N):
    X, Y, xs = L.make_grid(N, dev); errs = []
    for c, (ref, Xf, Yf) in zip(cases, refs):
        p = L.case_to_params(c, dev); inp, M, _ = L.channels_from_params(p, X, Y)
        with torch.no_grad(): T = L.hard_bc(m(inp), Dm, Dv)[0].cpu().numpy()
        T[M[0].cpu().numpy() < 0.5] = np.nan
        dx = 1 / (N - 1); margin = np.zeros((N, N), bool)
        for cc in c['cavities']: margin |= L.cav_phi_np(Xf, Yf, cc) < 2.5 * dx
        mk = (~np.isnan(T)) & (~np.isnan(ref)) & (~margin)
        if mk.sum() > 100: errs.append(np.sqrt(np.mean((T - ref)[mk] ** 2)) / (np.sqrt(np.mean(ref[mk] ** 2)) + 1e-9))
    return float(np.mean(errs))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=2500); ap.add_argument('--ntest', type=int, default=24); a = ap.parse_args()
    rows = []
    base = dict(N=128, modes=24, width=48, layers=4)
    sweeps = [('N', [96, 128, 160]), ('modes', [12, 24, 32]), ('width', [32, 48, 64])]
    cache = {}
    def get_test(N):
        if N not in cache: cache[N] = fixed_testset(a.ntest, N)
        return cache[N]
    for axis, vals in sweeps:
        for v in vals:
            cfg = dict(base); cfg[axis] = v
            t0 = time.time(); m, Dm, Dv = train_short(cfg['N'], cfg['modes'], cfg['width'], cfg['layers'], a.steps, seed=0)
            cases, refs = get_test(cfg['N']); rel = evaluate(m, Dm, Dv, cases, refs, cfg['N'])
            rows.append(dict(axis=axis, value=v, **cfg, relL2=rel, train_s=time.time() - t0))
            print(f'{axis}={v}: relL2={rel*100:.2f}%')
    # multi-seed headline
    seeds = []
    cases, refs = get_test(128)
    for s in range(3):
        m, Dm, Dv = train_short(128, 24, 48, 4, a.steps, seed=s); seeds.append(evaluate(m, Dm, Dv, cases, refs, 128))
    print(f'headline 3-seed relL2: {np.mean(seeds)*100:.2f}% ± {np.std(seeds)*100:.2f}%')
    rows.append(dict(axis='seed', value='0-2', N=128, modes=24, width=48, layers=4, relL2=np.mean(seeds), train_s=np.nan))
    df = pd.DataFrame(rows); df.to_csv('results/ablation.csv', index=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, (axis, vals) in enumerate(sweeps):
        d = df[df.axis == axis]; ax[k].plot(d.value, d.relL2 * 100, 'o-'); ax[k].set_xlabel(axis); ax[k].set_ylabel('relL2 [%]'); ax[k].set_title(f'vs {axis}'); ax[k].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('figures/F4_ablation.png', dpi=130); print('saved results/ablation.csv, figures/F4_ablation.png')

if __name__ == '__main__':
    main()
