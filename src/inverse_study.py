"""WP-E: hardened geometric inverse through the frozen PINO.
Recover single-cavity params (cx,cy,r,a,phi) from temperature sensors (interior grid / sparse /
top-only), with: noise study (sigma in K, many repetitions), multi-start + Tikhonov, UQ (ensemble
std), Fisher/sensitivity identifiability, and a sensor-placement comparison.
Outputs: results/inverse_noise.csv, figures F2 & F3."""
import os, argparse, time
import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import divino as L
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)  # output folders (relative to src/)
from fem import solve_fem, fem_to_grid

DT = 75.0  # K per non-dim unit (theta in [0,1] ~ [25,100] K)
PNAMES = ['cx', 'cy', 'r', 'a', 'phi']
PR = dict(cx=(0.30, 0.70), cy=(0.30, 0.70), r=(0.10, 0.22), a=(2.0, 8.0), phi=(0.0, np.pi / 2))

def sensor_idx(kind, N):
    if kind == 'top':
        j = np.arange(2, N - 2, 2); return (np.full_like(j, N - 1), j)
    if kind == 'grid':
        g = np.linspace(0.12, 0.88, 9); ii = (g * (N - 1)).astype(int)
        I, J = np.meshgrid(ii, ii); return (I.ravel(), J.ravel())
    if kind == 'sparse':
        rng = np.random.default_rng(7); pts = rng.uniform(0.12, 0.88, (16, 2))
        return ((pts[:, 1] * (N - 1)).astype(int), (pts[:, 0] * (N - 1)).astype(int))

def fem_obs(case, N, idx, noise_K, rng):
    m, th = solve_fem(case, hf=0.011); Tg, _, _ = fem_to_grid(m, th, N)
    v = Tg[idx[0], idx[1]]
    v = np.where(np.isnan(v), 0.0, v)
    if noise_K > 0: v = v + rng.normal(0, noise_K / DT, v.shape)
    return torch.tensor(v, dtype=torch.float32)

def build_params(vec, device):
    """vec: (5,) tensor in [0,1]; map to physical ranges (differentiable). Single cavity."""
    p = {}
    for i, k in enumerate(PNAMES):
        lo, hi = PR[k]; p[f'{k}1'] = (lo + (hi - lo) * torch.sigmoid(vec[i])).view(1)
    for k in PNAMES: p[f'{k}2'] = torch.tensor([0.5 if k in ('cx', 'cy') else (0.1 if k == 'r' else (4.0 if k == 'a' else 0.0))], device=device)
    p['use2'] = torch.tensor([0.0], device=device)
    p['Q0'] = torch.tensor([0.0], device=device); p['h'] = torch.tensor([0.0], device=device)
    return p

def recover(model, X, Y, Dmask, Dval, idx, obs, device, steps=300, lr=0.1, lam=1e-3, starts=4, seed=0):
    obs = obs.to(device); rng = np.random.default_rng(seed); best = None
    I = torch.tensor(idx[0], device=device); J = torch.tensor(idx[1], device=device)
    for st in range(starts):
        vec = torch.tensor(rng.normal(0, 1.0, 5), dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam([vec], lr=lr); traj = []
        for s in range(steps):
            p = build_params(vec, device)
            inp, _, _ = L.channels_from_params(p, X, Y)
            T = L.hard_bc(model(inp), Dmask, Dval)[0]
            pred = T[I, J]
            loss = torch.mean((pred - obs) ** 2) + lam * torch.mean(vec ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 10 == 0:
                with torch.no_grad():
                    pp = build_params(vec, device); traj.append([float(pp[f'{k}1']) for k in PNAMES] + [loss.item()])
        with torch.no_grad():
            pp = build_params(vec, device); fit = float(torch.mean((L.hard_bc(model(L.channels_from_params(pp, X, Y)[0]), Dmask, Dval)[0][I, J] - obs) ** 2))
        rec = {k: float(pp[f'{k}1']) for k in PNAMES}
        if best is None or fit < best[1]: best = (rec, fit, np.array(traj))
    return best

def fisher_svals(model, X, Y, Dmask, Dval, idx, case, device):
    """sensitivity (Jacobian) singular values of sensor obs w.r.t. the 5 params at the true case."""
    I = torch.tensor(idx[0], device=device); J = torch.tensor(idx[1], device=device)
    v0 = torch.zeros(5, device=device)
    c0 = case['cavities'][0]
    for i, k in enumerate(PNAMES):
        lo, hi = PR[k]; z = np.clip((c0[i] - lo) / (hi - lo), 1e-3, 1 - 1e-3); v0[i] = float(np.log(z / (1 - z)))
    v0.requires_grad_(True)
    p = build_params(v0, device); T = L.hard_bc(model(L.channels_from_params(p, X, Y)[0]), Dmask, Dval)[0]
    pred = T[I, J]
    Jac = torch.stack([torch.autograd.grad(pred[k], v0, retain_graph=True)[0] for k in range(len(pred))])  # (S,5)
    s = torch.linalg.svdvals(Jac).detach().cpu().numpy()
    return s / s.max()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/pino.pt')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--steps', type=int, default=250)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, _ = L.load_model(a.ckpt, dev); N = cfg['N']
    X, Y, xs = L.make_grid(N, dev); Dmask, Dval = L.dirichlet_fields(N, dev)
    truth = {'cavities': [(0.42, 0.55, 0.17, 5.0, 0.5)], 'Q0': 0.0, 'h': 0.0}
    c0 = truth['cavities'][0]
    noises = [0.0, 0.1, 0.5, 1.0, 2.0]; rows = []
    print('=== Table T4: inverse recovery (sensor=grid) bias+-std over reps, per param ===')
    for sg in noises:
        recs = []
        for rep in range(a.reps):
            rng = np.random.default_rng(1000 + rep)
            obs = fem_obs(truth, N, sensor_idx('grid', N), sg, rng)
            rec, fit, traj = recover(model, X, Y, Dmask, Dval, sensor_idx('grid', N), obs, dev, steps=a.steps, seed=rep)
            recs.append([rec[k] for k in PNAMES])
        recs = np.array(recs); mean = recs.mean(0); std = recs.std(0)
        for i, k in enumerate(PNAMES):
            rows.append(dict(sensor='grid', sigma_K=sg, param=k, true=c0[i], mean=mean[i], std=std[i], err=mean[i] - c0[i]))
        print(f'sigma={sg:>3} K | ' + '  '.join(f'{k}:{mean[i]:.3f}±{std[i]:.3f}(t{c0[i]:.2f})' for i, k in enumerate(PNAMES)))
    df = pd.DataFrame(rows); df.to_csv('results/inverse_noise.csv', index=False)

    # sensor comparison at sigma=0.5 K
    print('\n=== sensor placement (sigma=0.5 K) ===')
    sens_summary = {}
    for kind in ['top', 'sparse', 'grid']:
        recs = []
        for rep in range(max(8, a.reps // 2)):
            rng = np.random.default_rng(2000 + rep)
            obs = fem_obs(truth, N, sensor_idx(kind, N), 0.5, rng)
            rec, fit, _ = recover(model, X, Y, Dmask, Dval, sensor_idx(kind, N), obs, dev, steps=a.steps, seed=rep)
            recs.append([rec[k] for k in PNAMES])
        recs = np.array(recs); err = np.abs(recs.mean(0) - np.array(c0))
        sens_summary[kind] = err
        print(f'{kind:>7}: ' + '  '.join(f'{k}_err={err[i]:.3f}' for i, k in enumerate(PNAMES)))

    # Figure F2: convergence + recovered outlines (clean vs noisy)
    obs_c = fem_obs(truth, N, sensor_idx('grid', N), 0.0, np.random.default_rng(0))
    rec_c, _, traj = recover(model, X, Y, Dmask, Dval, sensor_idx('grid', N), obs_c, dev, steps=a.steps, seed=0)
    obs_n = fem_obs(truth, N, sensor_idx('grid', N), 1.0, np.random.default_rng(1))
    rec_n, _, _ = recover(model, X, Y, Dmask, Dval, sensor_idx('grid', N), obs_n, dev, steps=a.steps, seed=1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, k in enumerate(PNAMES):
        ax[0].plot(traj[:, i], label=k); ax[0].axhline(c0[i], ls='--', color=f'C{i}', alpha=0.5)
    ax[0].set_title('best-start convergence (clean)'); ax[0].set_xlabel('iter/10'); ax[0].legend(fontsize=8)
    ax[1].bar(noises, df[(df.sensor == 'grid') & (df.param == 'r')].err.abs(), width=0.12, label='|err r|')
    ax[1].bar(np.array(noises) + 0.13, df[(df.sensor == 'grid') & (df.param == 'cx')].err.abs(), width=0.12, label='|err cx|')
    ax[1].set_xlabel('noise sigma [K]'); ax[1].set_title('recovery error vs noise'); ax[1].legend()
    gx = L.cav_outline(c0); ax[2].plot(gx[:, 0], gx[:, 1], 'k-', lw=3, label='true')
    rc = (rec_c['cx'], rec_c['cy'], rec_c['r'], rec_c['a'], rec_c['phi']); g2 = L.cav_outline(rc)
    rn = (rec_n['cx'], rec_n['cy'], rec_n['r'], rec_n['a'], rec_n['phi']); g3 = L.cav_outline(rn)
    ax[2].plot(g2[:, 0], g2[:, 1], 'r--', lw=2, label='rec clean'); ax[2].plot(g3[:, 0], g3[:, 1], 'g:', lw=2, label='rec noisy(1K)')
    ax[2].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec='gray')); ax[2].set_aspect('equal'); ax[2].set_title('cavity recovery'); ax[2].legend()
    plt.tight_layout(); plt.savefig('figures/F2_inverse.png', dpi=130); print('saved figures/F2_inverse.png')

    # Figure F3: identifiability (Fisher sing. values) + sensor comparison
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for kind in ['top', 'sparse', 'grid']:
        s = fisher_svals(model, X, Y, Dmask, Dval, sensor_idx(kind, N), truth, dev)
        ax[0].semilogy(range(1, 6), s, 'o-', label=f'{kind}')
    ax[0].set_xlabel('mode'); ax[0].set_ylabel('norm. singular value'); ax[0].set_title('Sensitivity (identifiability) spectrum'); ax[0].legend()
    width = 0.25
    for j, kind in enumerate(['top', 'sparse', 'grid']):
        ax[1].bar(np.arange(5) + j * width, sens_summary[kind], width, label=kind)
    ax[1].set_xticks(np.arange(5) + width); ax[1].set_xticklabels(PNAMES); ax[1].set_ylabel('|error| @0.5K'); ax[1].set_title('sensor placement vs recovery error'); ax[1].legend()
    plt.tight_layout(); plt.savefig('figures/F3_identifiability.png', dpi=130); print('saved figures/F3_identifiability.png')

if __name__ == '__main__':
    main()
