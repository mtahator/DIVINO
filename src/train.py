"""WP-A/C: train the physics-informed FNO (PINO) over geometry (1-2 super-elliptic cavities with
center, size, exponent, rotation) + volumetric source Q0 + Robin coefficient h, via the Deep-Ritz
energy. Data-free. Saves a checkpoint."""
import os, sys, time, json, argparse
import numpy as np, torch
import divino as L

def get_cfg():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=128)
    p.add_argument('--modes', type=int, default=24)
    p.add_argument('--width', type=int, default=48)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--steps', type=int, default=int(os.environ.get('STEPS', 8000)))
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--two_cav_prob', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='checkpoints/pino.pt')
    return p.parse_args()

def main():
    c = get_cfg()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(c.seed); np.random.seed(c.seed); rng = np.random.default_rng(c.seed)
    X, Y, xs = L.make_grid(c.N, dev); Dmask, Dval = L.dirichlet_fields(c.N, dev); dx = 1.0 / (c.N - 1)
    model = L.FNO2d(cin=6, width=c.width, modes=c.modes, L=c.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=c.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c.steps, eta_min=c.lr / 10)
    hist = []; t0 = time.time()
    for it in range(c.steps):
        P, _ = L.sample_params(rng, c.batch, dev, two_cav_prob=c.two_cav_prob)
        inp, M, _ = L.channels_from_params(P, X, Y)
        theta = L.hard_bc(model(inp), Dmask, Dval)
        loss = L.ritz_energy(theta, M, P, dx)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); hist.append(loss.item())
        if it % max(1, c.steps // 12) == 0:
            print(f'[{it:5d}] energy={loss.item():.4e}', flush=True)
    print(f'trained {c.steps} steps in {time.time()-t0:.0f}s')
    os.makedirs(os.path.dirname(c.out), exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': vars(c), 'hist': hist}, c.out)
    print('saved', c.out)

if __name__ == '__main__':
    main()
