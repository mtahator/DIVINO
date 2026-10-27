"""Body-fitted P1 FEM reference (scikit-fem) for parametric heat conduction on the unit
square minus one/two super-elliptic cavities, with optional volumetric source Q and a
Robin/convective (or insulated) cavity boundary. Used as the ground truth for the PINO.

Physics (non-dimensional theta):  -div(grad theta) = Q  in Omega
  bottom y=0: Dirichlet theta = smoothstep ramp (hot),  left/right: theta = 0 (cold)
  top y=1: natural Neumann (insulated)
  cavity boundary: -d theta/dn = h (theta - Tinf)   (h=0 => insulated)
"""
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.tri as mtri
from skfem import (MeshTri, Basis, FacetBasis, ElementTriP1,
                   BilinearForm, LinearForm, asm, condense, solve)
from skfem.helpers import dot, grad

EPS_CORNER = 0.05
def smoothstep(e0, e1, t):
    t = np.clip((t - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)
def theta_bottom(x):
    return smoothstep(0, EPS_CORNER, x) * smoothstep(0, EPS_CORNER, 1 - x)

# ---------- super-ellipse geometry (center, size, exponent, rotation) ----------
def _rot(dx, dy, phi):
    c, s = np.cos(phi), np.sin(phi); return c * dx + s * dy, -s * dx + c * dy
def cav_phi(x, y, cav):
    """signed 'radius' level set: <0 inside, >0 outside (in rotated frame)."""
    cx, cy, r, a, phi = cav
    X, Y = _rot(x - cx, y - cy, phi)
    return (np.abs(X) ** a + np.abs(Y) ** a) ** (1.0 / a) - r
def cav_inside(x, y, cav): return cav_phi(x, y, cav) < 0
def cav_boundary(cav, n):
    cx, cy, r, a, phi = cav
    t = np.linspace(0, 2 * np.pi, n, endpoint=False); c, s = np.cos(t), np.sin(t)
    X = r * np.sign(c) * np.abs(c) ** (2 / a); Y = r * np.sign(s) * np.abs(s) ** (2 / a)
    cc, ss = np.cos(phi), np.sin(phi)
    return np.stack([cx + cc * X - ss * Y, cy + ss * X + cc * Y], 1)

# ---------- mesh: square minus cavities, boundary-conforming ----------
def build_mesh(cavities, hf=0.014, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(1 / hf)); e = np.linspace(0, 1, n + 1)
    z, o = np.zeros_like(e), np.ones_like(e)
    sq = np.concatenate([np.stack([e, z], 1), np.stack([e, o], 1),
                         np.stack([z, e], 1), np.stack([o, e], 1)])
    cavb = [cav_boundary(c, max(48, int(2 * np.pi * c[2] / hf))) for c in cavities]
    g = np.linspace(hf, 1 - hf, n - 1); gx, gy = np.meshgrid(g, g)
    ip = np.stack([gx.ravel(), gy.ravel()], 1) + (rng.random((gx.size, 2)) - 0.5) * hf * 0.5
    keep = np.ones(len(ip), bool)
    for c in cavities:
        keep &= cav_phi(ip[:, 0], ip[:, 1], c) > 0.6 * hf
    pts = np.unique(np.round(np.vstack([sq] + cavb + [ip[keep]]), 9), axis=0)
    tri = Delaunay(pts); cent = pts[tri.simplices].mean(1)
    ins = np.zeros(len(cent), bool)
    for c in cavities: ins |= cav_inside(cent[:, 0], cent[:, 1], c)
    return MeshTri(pts.T, tri.simplices[~ins].T.copy())

# ---------- solve ----------
def solve_fem(case, hf=0.014, seed=0):
    cav = case['cavities']; Q0 = case.get('Q0', 0.0); h = case.get('h', 0.0); Tinf = case.get('Tinf', 0.0)
    m = build_mesh(cav, hf, seed); e = ElementTriP1(); basis = Basis(m, e)
    K = asm(BilinearForm(lambda u, v, w: dot(grad(u), grad(v))), basis)
    b = asm(LinearForm(lambda v, w: Q0 * v), basis) if Q0 != 0 else np.zeros(basis.N)
    # cavity (Robin) facets = boundary facets not on the square edge
    bf = m.boundary_facets(); mid = m.p[:, m.facets[:, bf]].mean(1)
    onsq = (mid[0] < 1e-7) | (mid[0] > 1 - 1e-7) | (mid[1] < 1e-7) | (mid[1] > 1 - 1e-7)
    cav_facets = bf[~onsq]
    if h > 0 and cav_facets.size:
        fb = FacetBasis(m, e, facets=cav_facets)
        K = K + asm(BilinearForm(lambda u, v, w: h * u * v), fb)
        b = b + asm(LinearForm(lambda v, w: h * Tinf * v), fb)
    # Dirichlet walls: bottom ramp, left/right cold; top natural (insulated)
    x, y = m.p; tol = 1e-7
    bot = np.where(y < tol)[0]; sides = np.where((x < tol) | (x > 1 - tol))[0]
    D = np.unique(np.concatenate([bot, sides])); xb = np.zeros(basis.N)
    xb[bot] = theta_bottom(x[bot])              # sides already 0
    theta = solve(*condense(K, b, x=xb, D=D))
    return m, theta

def fem_to_grid(m, theta, N):
    """interpolate the P1 FEM field onto an N x N grid (NaN inside cavities)."""
    xs = np.linspace(0, 1, N); Xg, Yg = np.meshgrid(xs, xs)
    tri = mtri.Triangulation(m.p[0], m.p[1], m.t.T)
    interp = mtri.LinearTriInterpolator(tri, theta)
    Tg = interp(Xg.ravel(), Yg.ravel()).reshape(N, N)
    return np.asarray(Tg.filled(np.nan)), Xg, Yg

if __name__ == '__main__':
    import time
    case = {'cavities': [(0.5, 0.5, 0.2, 4.0, 0.0)], 'Q0': 0.0, 'h': 0.0}
    # self-convergence: two mesh densities
    fields = []
    for hf in (0.02, 0.013, 0.009):
        t0 = time.time(); m, th = solve_fem(case, hf=hf)
        Tg, X, Y = fem_to_grid(m, th, 200)
        fields.append(Tg); print(f'hf={hf}: nodes={m.p.shape[1]} tris={m.t.shape[1]} solve={time.time()-t0:.1f}s  T[0.5,0.99]={Tg[-2,100]:.4f}')
    a, b, c = fields
    msk = ~np.isnan(b) & ~np.isnan(c)
    rel_bc = np.sqrt(np.nanmean((b - c)[msk] ** 2)) / np.sqrt(np.nanmean(c[msk] ** 2))
    msk2 = ~np.isnan(a) & ~np.isnan(b)
    rel_ab = np.sqrt(np.nanmean((a - b)[msk2] ** 2)) / np.sqrt(np.nanmean(b[msk2] ** 2))
    print(f'self-convergence relL2: coarse->mid={rel_ab:.2e}  mid->fine={rel_bc:.2e}')
