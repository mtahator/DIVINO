"""Core library of DIVINO, the variational Fourier neural operator, with extended physics:
  -div(grad T) = Q0   on the unit square minus 1-2 super-elliptic cavities
  bottom Dirichlet (smoothed ramp), sides cold, top insulated (natural),
  cavity Robin:  -dT/dn = h (T - Tinf)   (h=0 => insulated, natural BC).

A "case" dict: {'cavities':[(cx,cy,r,a,phi),...], 'Q0':float, 'h':float, 'Tinf':float}.
The operator input channels are smooth & differentiable in the geometry/physics params,
so the same trained operator serves the geometric inverse.
"""
import numpy as np, torch, torch.nn as nn
import scipy.sparse as sp
import scipy.sparse.linalg as spla

CX0 = CY0 = 0.5
EPS_CORNER = 0.05
TAU = 0.012
Tinf_DEFAULT = 0.0

# ---------------- numpy geometry ----------------
def smoothstep(e0, e1, t): t = np.clip((t - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)
def theta_bottom(x): return smoothstep(0, EPS_CORNER, x) * smoothstep(0, EPS_CORNER, 1 - x)
def _rot_np(dx, dy, phi): c, s = np.cos(phi), np.sin(phi); return c * dx + s * dy, -s * dx + c * dy
def cav_phi_np(x, y, cav):
    cx, cy, r, a, phi = cav; X, Y = _rot_np(x - cx, y - cy, phi)
    return (np.abs(X) ** a + np.abs(Y) ** a) ** (1.0 / a) - r
def cavs_inside_np(x, y, cavs):
    m = np.zeros(np.shape(x), bool)
    for c in cavs: m |= cav_phi_np(x, y, c) < 0
    return m
def cav_outline(cav, n=400):
    cx, cy, r, a, phi = cav; t = np.linspace(0, 2 * np.pi, n); c, s = np.cos(t), np.sin(t)
    X = r * np.sign(c) * np.abs(c) ** (2 / a); Y = r * np.sign(s) * np.abs(s) ** (2 / a)
    cc, ss = np.cos(phi), np.sin(phi)
    return np.stack([cx + cc * X - ss * Y, cy + ss * X + cc * Y], 1)
def superellipse_perimeter_np(r, a, n=512):
    t = np.linspace(0, 2 * np.pi, n + 1); ct, st = np.cos(t), np.sin(t)
    X = r * np.sign(ct) * np.abs(ct) ** (2 / a); Y = r * np.sign(st) * np.abs(st) ** (2 / a)
    return np.sqrt(np.diff(X) ** 2 + np.diff(Y) ** 2).sum()

# ---------------- parameter sampling ----------------
RANGES = dict(r=(0.10, 0.22), a=(2.0, 8.0), cx=(0.30, 0.70), cy=(0.30, 0.70),
              phi=(0.0, np.pi / 2), Q0=(0.0, 3.0), h=(0.0, 8.0))
def _u(rng, k, n): lo, hi = RANGES[k]; return rng.uniform(lo, hi, n)
def sample_case(rng, two_cav_prob=0.5, ood=0.0):
    """Return a case dict. ood>0 widens ranges by that fraction."""
    def u(k):
        lo, hi = RANGES[k]; m = (lo + hi) / 2; half = (hi - lo) / 2 * (1 + ood)
        return float(np.clip(rng.uniform(m - half, m + half), *(RANGES[k] if ood == 0 else (lo - half, hi + half))))
    cavs = []
    ncav = 2 if rng.random() < two_cav_prob else 1
    for _ in range(ncav):
        for _try in range(40):
            cav = (u('cx'), u('cy'), u('r'), u('a'), u('phi'))
            if not cavs or all((cav[0] - c[0]) ** 2 + (cav[1] - c[1]) ** 2 > (cav[2] + c[2] + 0.05) ** 2 for c in cavs):
                # keep inside domain
                if 0.12 < cav[0] < 0.88 and 0.12 < cav[1] < 0.88 and cav[2] < min(cav[0], 1 - cav[0], cav[1], 1 - cav[1]) - 0.03:
                    cavs.append(cav); break
        else:
            break
    return {'cavities': cavs, 'Q0': u('Q0'), 'h': u('h'), 'Tinf': Tinf_DEFAULT}

# ---------------- torch input channels (differentiable in params) ----------------
def make_grid(N, device):
    xs = np.linspace(0, 1, N); Xg, Yg = np.meshgrid(xs, xs)
    return (torch.tensor(Xg, dtype=torch.float32, device=device),
            torch.tensor(Yg, dtype=torch.float32, device=device), xs)

def cav_phi_torch(X, Y, cav):
    cx, cy, r, a, phi = cav
    c, s = torch.cos(phi), torch.sin(phi)
    dx, dy = X - cx, Y - cy
    Xr, Yr = c * dx + s * dy, -s * dx + c * dy
    return (torch.abs(Xr) ** a + torch.abs(Yr) ** a) ** (1.0 / a) - r

def channels_from_params(params, X, Y):
    """params: dict of (B,) tensors: cx1,cy1,r1,a1,phi1, [cx2..], use2 (0/1), Q0, h.
    Returns FNO input (B,6,N,N) [x,y,smask,levelset,Qfield,h] and binary fluid mask M (B,N,N)."""
    B = params['r1'].shape[0]; N = X.shape[0]; dev = X.device
    Xb = X[None].expand(B, N, N); Yb = Y[None].expand(B, N, N)
    def phi_of(idx):
        cav = (params[f'cx{idx}'][:, None, None], params[f'cy{idx}'][:, None, None],
               params[f'r{idx}'][:, None, None], params[f'a{idx}'][:, None, None],
               params[f'phi{idx}'][:, None, None])
        return cav_phi_torch(Xb, Yb, cav)
    phi1 = phi_of(1)
    if 'r2' in params:
        phi2 = phi_of(2)
        use2 = params['use2'][:, None, None]
        # where cavity-2 is active, take min; else just phi1 (push phi2 to +inf)
        phi2 = torch.where(use2 > 0.5, phi2, torch.full_like(phi2, 10.0))
        phi = torch.minimum(phi1, phi2)
    else:
        phi = phi1
    smask = torch.sigmoid(phi / TAU)
    Qf = params['Q0'][:, None, None].expand(B, N, N)
    hf = params['h'][:, None, None].expand(B, N, N)
    inp = torch.stack([Xb * 2 - 1, Yb * 2 - 1, smask, torch.tanh(phi / 0.1), Qf / 3.0, hf / 8.0], 1)
    M = (phi > 0).float()
    return inp, M, phi

def case_to_params(case, device):
    """one case dict -> params with B=1 tensors (non-differentiable helper for eval)."""
    cavs = case['cavities']; c1 = cavs[0]
    p = {f'{k}1': torch.tensor([v], dtype=torch.float32, device=device)
         for k, v in zip(['cx', 'cy', 'r', 'a', 'phi'], c1)}
    if len(cavs) > 1:
        c2 = cavs[1]
        p.update({f'{k}2': torch.tensor([v], dtype=torch.float32, device=device)
                  for k, v in zip(['cx', 'cy', 'r', 'a', 'phi'], c2)})
        p['use2'] = torch.tensor([1.0], device=device)
    else:
        p.update({f'{k}2': torch.tensor([0.5], dtype=torch.float32, device=device) for k in ['cx', 'cy', 'r', 'a', 'phi']})
        p['r2'] = torch.tensor([0.10], device=device); p['a2'] = torch.tensor([4.0], device=device)
        p['use2'] = torch.tensor([0.0], device=device)
    p['Q0'] = torch.tensor([case.get('Q0', 0.0)], dtype=torch.float32, device=device)
    p['h'] = torch.tensor([case.get('h', 0.0)], dtype=torch.float32, device=device)
    return p

def sample_params(rng, B, device, two_cav_prob=0.5, ood=0.0):
    """vectorized batch of random params (always carries cavity-2 slots; use2 gates it)."""
    cases = [sample_case(rng, two_cav_prob, ood) for _ in range(B)]
    P = {}
    for k in ['cx', 'cy', 'r', 'a', 'phi']:
        P[f'{k}1'] = torch.tensor([c['cavities'][0][['cx', 'cy', 'r', 'a', 'phi'].index(k)] for c in cases], dtype=torch.float32, device=device)
    for k in ['cx', 'cy', 'r', 'a', 'phi']:
        P[f'{k}2'] = torch.tensor([(c['cavities'][1][['cx', 'cy', 'r', 'a', 'phi'].index(k)] if len(c['cavities']) > 1 else (0.5 if k in 'cxcy' else (0.1 if k == 'r' else (4.0 if k == 'a' else 0.0)))) for c in cases], dtype=torch.float32, device=device)
    P['use2'] = torch.tensor([1.0 if len(c['cavities']) > 1 else 0.0 for c in cases], dtype=torch.float32, device=device)
    P['Q0'] = torch.tensor([c['Q0'] for c in cases], dtype=torch.float32, device=device)
    P['h'] = torch.tensor([c['h'] for c in cases], dtype=torch.float32, device=device)
    return P, cases

# ---------------- Deep-Ritz energy with source + Robin ----------------
def hard_bc(raw, Dmask, Dval): return raw * (1 - Dmask) + Dval

def dirichlet_fields(N, device):
    xs = np.linspace(0, 1, N); ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    Dmask = torch.tensor(((ii == 0) | (jj == 0) | (jj == N - 1)).astype('f4'), device=device)
    Dval = torch.zeros(N, N, device=device); Dval[0, :] = torch.tensor(theta_bottom(xs), dtype=torch.float32, device=device)
    return Dmask, Dval

def superellipse_perimeter_t(r, a, n=512):
    """analytic (polygon) perimeter of |X|^a+|Y|^a=r^a; r,a:(B,) -> (B,)."""
    t = torch.linspace(0, 2 * np.pi, n + 1, device=r.device)
    ct, st = torch.cos(t), torch.sin(t)
    X = r[:, None] * torch.sign(ct)[None] * torch.abs(ct)[None] ** (2.0 / a[:, None])
    Y = r[:, None] * torch.sign(st)[None] * torch.abs(st)[None] ** (2.0 / a[:, None])
    return torch.sqrt((X[:, 1:] - X[:, :-1]) ** 2 + (Y[:, 1:] - Y[:, :-1]) ** 2).sum(1)

def ritz_energy(theta, M, params, dx, Tinf=0.0):
    """theta (B,N,N). Deep-Ritz energy per sample (mean over batch).
    The Robin boundary term uses a per-sample perimeter correction (analytic/staircase) so the
    Cartesian staircase does not over-count the cavity boundary length (WP-G)."""
    B = theta.shape[0]
    dh = theta[:, :, 1:] - theta[:, :, :-1]; Mh = M[:, :, 1:] * M[:, :, :-1]
    dv = theta[:, 1:, :] - theta[:, :-1, :]; Mv = M[:, 1:, :] * M[:, :-1, :]
    E = 0.5 * (Mh * dh ** 2).sum() + 0.5 * (Mv * dv ** 2).sum()              # stiffness
    Q0 = params['Q0'][:, None, None]
    E = E - (dx * dx) * (Q0 * theta * M).sum()                              # source  -∫Q T
    h = params['h'][:, None, None]
    if float(h.abs().max()) > 0:
        solid = 1.0 - M
        faces = torch.zeros_like(M)
        faces[:, :, :-1] += solid[:, :, 1:]; faces[:, :, 1:] += solid[:, :, :-1]
        faces[:, :-1, :] += solid[:, 1:, :]; faces[:, 1:, :] += solid[:, :-1, :]
        band = M * (faces > 0).float()
        Ls = (faces * band).sum(dim=(1, 2)) * dx + 1e-9                     # staircase perimeter
        Lt = superellipse_perimeter_t(params['r1'], params['a1'])
        if 'r2' in params:
            Lt = Lt + params['use2'] * superellipse_perimeter_t(params['r2'], params['a2'])
        scale = (Lt / Ls).clamp(0.3, 1.0)[:, None, None]                    # correct length over-count
        E = E + 0.5 * (h * dx * scale * faces * band * (theta - Tinf) ** 2).sum()
    return E / B

# ---------------- FNO ----------------
class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__(); self.cout, self.m1, self.m2 = cout, m1, m2; s = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
    def forward(self, x):
        B, C, H, W = x.shape; xft = torch.fft.rfft2(x)
        out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.m1, H // 2); m2 = min(self.m2, W // 2 + 1)
        out[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy', xft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum('bixy,ioxy->boxy', xft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H, W))
class FNO2d(nn.Module):
    def __init__(self, cin=6, width=48, modes=24, L=4):
        super().__init__(); self.fc0 = nn.Linear(cin, width)
        self.sp = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(L)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(L)])
        self.fc1 = nn.Linear(width, 128); self.fc2 = nn.Linear(128, 1); self.act = nn.GELU()
    def forward(self, x):
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for sp, w in zip(self.sp, self.w): x = self.act(sp(x) + w(x))
        x = self.act(self.fc1(x.permute(0, 2, 3, 1)))
        return self.fc2(x).squeeze(-1)

# ---------------- direct grid solver (quadratic energy -> linear system); physics check & fast grid ref ----------------
def direct_grid_solve(case, N, device='cpu'):
    X, Y, xs = make_grid(N, device); p = case_to_params(case, device)
    _, M, _ = channels_from_params(p, X, Y)
    M = M[0].cpu().numpy().astype(bool); dx = 1.0 / (N - 1)
    Dmask, Dval = dirichlet_fields(N, device); Dm = Dmask.cpu().numpy().astype(bool); Dv = Dval.cpu().numpy()
    idx = -np.ones((N, N), int); free = M & ~Dm; idx[free] = np.arange(free.sum())
    nf = int(free.sum()); rows, cols, data = [], [], []; b = np.zeros(nf)
    Q0 = case.get('Q0', 0.0); h = case.get('h', 0.0); Tinf = case.get('Tinf', 0.0)
    # per-case Robin perimeter correction (staircase over-counts boundary length)
    rscale = 1.0
    if h > 0:
        solid = ~M; faces = np.zeros((N, N))
        faces[:, :-1] += solid[:, 1:]; faces[:, 1:] += solid[:, :-1]
        faces[:-1, :] += solid[1:, :]; faces[1:, :] += solid[:-1, :]
        Ls = faces[M & (faces > 0)].sum() * dx
        Lt = sum(superellipse_perimeter_np(c[2], c[3]) for c in case['cavities'])
        rscale = float(np.clip(Lt / (Ls + 1e-9), 0.3, 1.0))
    h = h * rscale
    def add(i, j, v): rows.append(i); cols.append(j); data.append(v)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(N):
        for j in range(N):
            if not free[i, j]: continue
            k = idx[i, j]; diag = 0.0; nsolid = 0
            for di, dj in nbrs:
                ni, nj = i + di, j + dj
                if ni < 0 or nj < 0 or ni >= N or nj >= N or not M[ni, nj]:
                    if (ni < 0 or nj < 0 or ni >= N or nj >= N) is False and not M[ni, nj]:
                        nsolid += 1
                    continue
                diag += 1.0
                if Dm[ni, nj]: b[k] += Dv[ni, nj]
                elif free[ni, nj]: add(k, idx[ni, nj], -1.0)
            add(k, k, diag)
            b[k] += dx * dx * Q0
            if h > 0 and nsolid > 0:
                add(k, k, h * dx * nsolid); b[k] += h * dx * nsolid * Tinf
    A = sp.csr_matrix((data, (rows, cols)), shape=(nf, nf))
    sol = spla.spsolve(A, b); T = np.full((N, N), np.nan); T[free] = sol
    for i, j in zip(*np.where(Dm & M)): T[i, j] = Dv[i, j]
    return T, xs

# ---------------- eval helpers ----------------
def load_model(ckpt, device):
    d = torch.load(ckpt, map_location=device, weights_only=False); cfg = d['cfg']
    m = FNO2d(cin=6, width=cfg['width'], modes=cfg['modes'], L=cfg['layers']).to(device)
    m.load_state_dict(d['model']); m.eval()
    return m, cfg, d.get('hist', [])

def fno_eval(model, case, N, device):
    X, Y, xs = make_grid(N, device); Dmask, Dval = dirichlet_fields(N, device)
    p = case_to_params(case, device); inp, M, _ = channels_from_params(p, X, Y)
    with torch.no_grad():
        th = hard_bc(model(inp), Dmask, Dval)[0].cpu().numpy()
    th[M[0].cpu().numpy() < 0.5] = np.nan
    return th, xs
