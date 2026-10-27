"""WP-D: baselines & efficiency.
 B2 strong-form PINO (residual loss) vs our variational PINO -> validates kappa vs kappa^2.
 B3 data-driven FNO (supervised on FEM) -> same architecture, needs data.
 B1 MLP-PINN (per-geometry, single instance) -> the 'no operator' baseline + its cost.
 B4 FEM solve timing + FNO inference timing -> speedup (T3).
Outputs Tables T2 (accuracy vs cost) and T3 (speedup) to stdout + results/baselines.csv."""
import os, time, argparse
import numpy as np, torch, torch.nn as nn, torch.autograd as autograd, pandas as pd
import divino as L
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)  # output folders (relative to src/)
from fem import solve_fem, fem_to_grid

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def testset(n, N, seed=321):
    rng = np.random.default_rng(seed); cases = [L.sample_case(rng, two_cav_prob=0.4) for _ in range(n)]
    refs = []
    for c in cases:
        m, th = solve_fem(c, hf=0.012); Tg, X, Y = fem_to_grid(m, th, N); refs.append(Tg)
    return cases, refs, X, Y

def relL2(Tf, Tref, case, N):
    dx = 1.0 / (N - 1); margin = np.zeros((N, N), bool)
    for c in case['cavities']: margin |= L.cav_phi_np(*np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N)), c) < 2.5*dx
    mk = (~np.isnan(Tf)) & (~np.isnan(Tref)) & (~margin)
    return float(np.sqrt(np.mean((Tf-Tref)[mk]**2)) / (np.sqrt(np.mean(Tref[mk]**2))+1e-9))

def eval_model(model, Dmask, Dval, cases, refs, X, Y, N):
    errs = []
    for c, ref in zip(cases, refs):
        p = L.case_to_params(c, dev); inp, M, _ = L.channels_from_params(p, X, Y)
        with torch.no_grad(): T = L.hard_bc(model(inp), Dmask, Dval)[0].cpu().numpy()
        T[M[0].cpu().numpy() < 0.5] = np.nan; errs.append(relL2(T, ref, c, N))
    return np.mean(errs), np.percentile(errs, 95)

# ---- strong-form residual (kappa^2) ----
def strong_residual(theta, M, P, dx):
    c = theta[:,1:-1,1:-1]; up=theta[:,:-2,1:-1]; dn=theta[:,2:,1:-1]; lf=theta[:,1:-1,:-2]; rt=theta[:,1:-1,2:]
    Mc=M[:,1:-1,1:-1]; Mu=M[:,:-2,1:-1]; Md=M[:,2:,1:-1]; Ml=M[:,1:-1,:-2]; Mr=M[:,1:-1,2:]
    cnt=Mu+Md+Ml+Mr; lap=cnt*c-(Mu*up+Md*dn+Ml*lf+Mr*rt)
    Q0=P['Q0'][:,None,None]; res = lap - dx*dx*Q0       # graph-Laplacian = dx^2 Q
    w=Mc*(cnt>0).float()
    L_in = torch.sum((res*w)**2) / w.sum().clamp(min=1)
    ct=theta[:,-1,1:-1]; bl=theta[:,-2,1:-1]; lt=theta[:,-1,:-2]; rg=theta[:,-1,2:]
    L_top=torch.mean((4*ct-2*bl-lt-rg)**2)
    return L_in + 0.5*L_top

def train_fno(loss_kind, steps, N, seed=0):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    X,Y,xs=L.make_grid(N,dev); Dmask,Dval=L.dirichlet_fields(N,dev); dx=1/(N-1)
    m=L.FNO2d(cin=6,width=40,modes=20,L=4).to(dev); opt=torch.optim.Adam(m.parameters(),2e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=2e-4); t0=time.time()
    for it in range(steps):
        P,_=L.sample_params(rng,12,dev,two_cav_prob=0.3)
        inp,M,_=L.channels_from_params(P,X,Y); th=L.hard_bc(m(inp),Dmask,Dval)
        loss = L.ritz_energy(th,M,P,dx) if loss_kind=='energy' else strong_residual(th,M,P,dx)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m, Dmask, Dval, time.time()-t0

def train_datadriven(steps, N, ndata=150, seed=0):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    X,Y,xs=L.make_grid(N,dev); Dmask,Dval=L.dirichlet_fields(N,dev)
    cases=[L.sample_case(rng,0.3) for _ in range(ndata)]; ins=[]; tgt=[]
    for c in cases:
        p=L.case_to_params(c,dev); inp,Mt,_=L.channels_from_params(p,X,Y)
        mm,th=solve_fem(c,hf=0.013); Tg,_,_=fem_to_grid(mm,th,N)
        Tg=np.nan_to_num(Tg,nan=0.0); ins.append(inp[0].cpu()); tgt.append(torch.tensor(Tg,dtype=torch.float32))
    ins=torch.stack(ins).to(dev); tgt=torch.stack(tgt).to(dev)
    m=L.FNO2d(cin=6,width=40,modes=20,L=4).to(dev); opt=torch.optim.Adam(m.parameters(),2e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=2e-4); t0=time.time()
    for it in range(steps):
        bi=rng.integers(0,ndata,12); pred=L.hard_bc(m(ins[bi]),Dmask,Dval)
        loss=torch.mean((pred-tgt[bi])**2); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m, Dmask, Dval, time.time()-t0

def mlp_pinn_one(case, N, epochs=1500):
    """per-geometry strong-form MLP PINN on a single case (insulated); return relL2 + time."""
    X,Y,xs=L.make_grid(N,dev); Dmask,Dval=L.dirichlet_fields(N,dev)
    p=L.case_to_params(case,dev); _,M,_=L.channels_from_params(p,X,Y); Mnp=M[0].cpu().numpy()>0.5
    net=nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1)).to(dev)
    opt=torch.optim.Adam(net.parameters(),2e-3); cav=case['cavities']
    fluid_pts=np.stack(np.where(Mnp),1)/(N-1); fluid_pts=fluid_pts[:, ::-1].copy()
    t0=time.time()
    for e in range(epochs):
        ci=np.random.randint(0,len(fluid_pts),1500); xy=torch.tensor(fluid_pts[ci],dtype=torch.float32,device=dev,requires_grad=True)
        u=net(xy); g=autograd.grad(u,xy,torch.ones_like(u),create_graph=True)[0]
        uxx=autograd.grad(g[:,0],xy,torch.ones_like(g[:,0]),create_graph=True)[0][:,0]
        uyy=autograd.grad(g[:,1],xy,torch.ones_like(g[:,1]),create_graph=True)[0][:,1]
        Lpde=torch.mean((uxx+uyy)**2)
        xb=torch.rand(400,device=dev); bot=torch.stack([xb,torch.zeros_like(xb)],1)
        tb=torch.tensor(L.theta_bottom(xb.cpu().numpy()),dtype=torch.float32,device=dev)
        ld=torch.mean((net(bot).squeeze()-tb)**2)
        yl=torch.rand(400,device=dev)
        ld=ld+torch.mean(net(torch.stack([torch.zeros_like(yl),yl],1))**2)+torch.mean(net(torch.stack([torch.ones_like(yl),yl],1))**2)
        loss=Lpde+30*ld; opt.zero_grad(); loss.backward(); opt.step()
    tt=time.time()-t0
    flat=torch.tensor(np.stack(np.meshgrid(xs,xs),-1).reshape(-1,2),dtype=torch.float32,device=dev)
    with torch.no_grad(): T=net(flat).cpu().numpy().reshape(N,N)
    T[~Mnp]=np.nan
    mm,th=solve_fem(case,hf=0.012); ref,_,_=fem_to_grid(mm,th,N)
    return relL2(T,ref,case,N), tt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',default='checkpoints/pino.pt')
    ap.add_argument('--ntest',type=int,default=40); ap.add_argument('--steps',type=int,default=3000); a=ap.parse_args()
    model,cfg,_=L.load_model(a.ckpt,dev); N=cfg['N']
    cases,refs,_,_=testset(a.ntest,N)
    X,Y,_=L.make_grid(N,dev)
    Dmask,Dval=L.dirichlet_fields(N,dev)
    rows=[]
    # ours (the trained checkpoint)
    me,p95=eval_model(model,Dmask,Dval,cases,refs,X,Y,N)
    # FNO inference timing
    p=L.case_to_params(cases[0],dev); inp,_,_=L.channels_from_params(p,X,Y)
    torch.cuda.synchronize() if dev.type=='cuda' else None; t0=time.time()
    for _ in range(50):
        with torch.no_grad(): _=model(inp)
    torch.cuda.synchronize() if dev.type=='cuda' else None; t_inf=(time.time()-t0)/50
    rows.append(dict(method='This work (variational PINO)',relL2_mean=me,relL2_p95=p95,train_s=np.nan,infer_s=t_inf,params=sum(x.numel() for x in model.parameters())))
    print(f'[ours] relL2 mean={me:.3e} p95={p95:.3e} infer={t_inf*1e3:.2f} ms')
    # B2 strong-form
    m2,Dm2,Dv2,tt2=train_fno('strong',a.steps,N); e2,p2=eval_model(m2,Dm2,Dv2,cases,refs,X,Y,N)
    rows.append(dict(method='B2 strong-form PINO',relL2_mean=e2,relL2_p95=p2,train_s=tt2,infer_s=t_inf,params=sum(x.numel() for x in m2.parameters())))
    print(f'[B2 strong-form] relL2 mean={e2:.3e}  (vs ours {me:.3e}) train={tt2:.0f}s')
    # B3 data-driven
    m3,Dm3,Dv3,tt3=train_datadriven(a.steps,N); e3,p3=eval_model(m3,Dm3,Dv3,cases,refs,X,Y,N)
    rows.append(dict(method='B3 data-driven FNO',relL2_mean=e3,relL2_p95=p3,train_s=tt3,infer_s=t_inf,params=sum(x.numel() for x in m3.parameters())))
    print(f'[B3 data-driven] relL2 mean={e3:.3e} train={tt3:.0f}s (incl FEM data)')
    # B1 MLP-PINN (per geometry) on 2 cases
    b1=[mlp_pinn_one(cases[i],N,epochs=1200) for i in range(2)]; e1=np.mean([x[0] for x in b1]); t1=np.mean([x[1] for x in b1])
    rows.append(dict(method='B1 MLP-PINN (per-geometry)',relL2_mean=e1,relL2_p95=np.nan,train_s=t1,infer_s=t1,params=net_params()))
    print(f'[B1 MLP-PINN] relL2 mean={e1:.3e}  cost/geometry={t1:.0f}s')
    # B4 FEM timing
    nt=min(5,len(cases)); t0=time.time(); [solve_fem(cases[i],hf=0.012) for i in range(nt)]; t_fem=(time.time()-t0)/nt
    print(f'[B4 FEM] {t_fem*1e3:.0f} ms/solve')
    df=pd.DataFrame(rows); df.to_csv('results/baselines.csv',index=False)
    print('\n=== Table T3: speedup (inference vs FEM) ===')
    print(f'FEM solve: {t_fem*1e3:.0f} ms | FNO inference: {t_inf*1e3:.2f} ms | speedup ~ {t_fem/t_inf:.0f}x per query')
    print('saved results/baselines.csv')

def net_params(): return 64*2+64+64*64+64+64*64+64+64+1

if __name__=='__main__':
    main()
