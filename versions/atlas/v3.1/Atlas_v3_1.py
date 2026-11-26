"""Atlas v3.1: independent residual physical downscaling."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import io, zipfile, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
NORM=np.array([25.360,2.0652,2.6776,2.0876],dtype='f4')
CUTS=[('Mayo','2024050900','2024051600','2024060100'),('Junio I','2024052500','2024060100','2024061600'),('Junio II','2024060900','2024061600','2024070100'),('Abril','2024040900','2024041600','2024050100')]
def load(root):
    root=Path(root)
    if root.is_file():
        with zipfile.ZipFile(root) as z:
            def read(n):
                with np.load(io.BytesIO(z.read(n))) as f: return dict(f)
            return read('train.npz'),read('static.npz')
    if not (root/'train.npz').exists():
        matches=[p.parent for p in root.rglob('train.npz') if (p.parent/'static.npz').exists() and (p.parent/'test.npz').exists()]
        if len(matches)!=1: raise ValueError(f'Dataset ambiguo: {matches}')
        root=matches[0]
    def read(n):
        with np.load(root/n) as f: return dict(f)
    return read('train.npz'),read('static.npz')
def windows(x,t):
    dates=pd.to_datetime(t,format='%Y%m%d%H'); lookup={d:i for i,d in enumerate(dates)}
    wanted=[[d-pd.Timedelta(hours=k) for k in (2,1,0)] for d in dates]
    idx=np.array([[lookup.get(q,i) for q in row] for i,row in enumerate(wanted)])
    mask=np.array([[q in lookup for q in row] for row in wanted],dtype='f4')
    return x[idx],mask
def fields(w,lat,lon):
    rh,t,u,v=np.moveaxis(w,-1,0); c=t-273.15
    gamma=np.log(np.clip(rh,.1,100)/100)+17.625*c/(243.04+c)
    dew=243.04*gamma/(17.625-gamma)
    e=6.1094*np.exp(17.625*c/(243.04+c))*rh/100
    dy=np.deg2rad(float(np.diff(lat).mean()))*6371000
    dx=np.deg2rad(float(np.diff(lon).mean()))*6371000*np.cos(np.deg2rad(lat))[None,None,:,None]
    def grad(a): return np.gradient(a,axis=-1)/dx,np.gradient(a,axis=-2)/dy
    ux,uy=grad(u); vx,vy=grad(v); tx,ty=grad(t); ex,ey=grad(e)
    return np.stack([rh,t,u,v,c-dew,e,np.hypot(u,v),(ux+vy)*1e5,(vx-uy)*1e5,(ux-vy)*1e5,(vx+uy)*1e5,-3600*(u*tx+v*ty),-3600*(u*ex+v*ey),np.hypot(tx,ty)*1e5,np.hypot(ex,ey)*1e5],axis=-1).astype('f4')
def features(w,physics,lat,lon):
    a=fields(w,lat,lon) if physics else w
    return np.concatenate([a[:,2],a[:,2]-a[:,1],a[:,2]-a[:,0]],axis=-1).reshape(len(w),-1)
def climate(a,h,tr):
    return np.stack([a[tr[h[tr]==i]].mean(0) if np.any(h[tr]==i) else a[tr].mean(0) for i in range(24)])
def metric(p,y):
    p=p.copy(); p[...,0]=p[...,0].clip(0,100)
    rm=np.sqrt(np.mean((p.astype('f8')-y)**2,axis=(0,1,2)))
    return float(1-(rm/NORM).mean()),rm

import math, json, copy
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import nn
from torch.nn import functional as F

BASE={'Mayo':.4599006,'Junio I':.4604175,'Junio II':.4650797440837502,'Abril':.5584575773496497}

def geography(s):
    lat,lon=np.meshgrid(s['wrf_lats'],s['wrf_lons'],indexing='ij'); dem=s['dem'].astype('f4')
    dy=np.deg2rad(float(np.diff(s['wrf_lats']).mean()))*6371000
    dx=np.deg2rad(float(np.diff(s['wrf_lons']).mean()))*6371000*np.cos(np.deg2rad(lat))
    sx=np.gradient(dem,axis=1)/dx; sy=np.gradient(dem,axis=0)/dy
    from scipy.ndimage import gaussian_filter
    relief=[(dem-gaussian_filter(dem,k))/500 for k in (1,3,7)]
    curvature=(np.gradient(sx,axis=1)/dx+np.gradient(sy,axis=0)/dy)*1000
    # Raw slopes retained separately for physically interpretable wind projection.
    static=np.stack([dem/3000,sx,sy,np.hypot(sx,sy),*relief,curvature,(lat-lat.mean()),(lon-lon.mean())]).astype('f4')
    gx=2*(lon-s['gfs_lons'][0])/(s['gfs_lons'][-1]-s['gfs_lons'][0])-1
    gy=2*(lat-s['gfs_lats'][0])/(s['gfs_lats'][-1]-s['gfs_lats'][0])-1
    grid=np.stack([gx,gy],-1)[None].astype('f4')
    return static[None],grid

def prepare(X,T,s):
    w,mask=windows(X,T); d=fields(w,s['gfs_lats'],s['gfs_lons'])
    x=np.concatenate([d[:,2],d[:,1],d[:,0]],axis=-1).transpose(0,3,1,2).copy()
    dates=pd.to_datetime(T,format='%Y%m%d%H'); h=dates.hour.to_numpy(); day=dates.dayofyear.to_numpy()
    a=h*2*np.pi/24; decl=.409*np.sin(2*np.pi*(day-81)/365.25)
    # Solar declination and UTC hour; longitude is applied inside the decoder.
    t=np.column_stack([np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a),np.sin(decl),np.cos(decl),mask]).astype('f4')
    return x,t,h

class Block(nn.Module):
    def __init__(self,c,d=1):
        super().__init__(); self.f=nn.Sequential(nn.Conv2d(c,c,3,padding=d,dilation=d),nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c))
    def forward(self,x): return F.silu(x+.5*self.f(x))

class ConvLSTM(nn.Module):
    def __init__(self,w):
        super().__init__(); self.gates=nn.Conv2d(2*w,4*w,3,padding=1)
    def forward(self,x,h,c):
        i,f,o,g=self.gates(torch.cat([x,h],1)).chunk(4,1)
        c=(f+1).sigmoid()*c+i.sigmoid()*g.tanh(); return o.sigmoid()*c.tanh(),c

class TerrainFlow(nn.Module):
    def __init__(self,static,grid,lat,lon,mu,sd,width=48,physics=True):
        super().__init__(); self.physics=physics; self.width=width
        for name,a in [('geo',static),('grid',grid),('latitude',np.deg2rad(lat)[None,None,:,None]),('longitude',np.deg2rad(lon)[None,None,None,:]),('mu',mu[None,:4,None,None]),('sd',sd[None,:4,None,None])]:
            self.register_buffer(name,torch.from_numpy(np.asarray(a,dtype='f4')))
        self.stem=nn.Sequential(nn.Conv2d(24,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU()); self.temporal=ConvLSTM(width); self.context=Block(width,2)
        self.pos=nn.Parameter(torch.zeros(1,17*13,width)); nn.init.normal_(self.pos,std=.02)
        self.attn=nn.MultiheadAttention(width,4,batch_first=True,dropout=.05); self.norm=nn.LayerNorm(width)
        self.geoenc=nn.Sequential(nn.Conv2d(10,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,24,3,padding=1),nn.SiLU())
        # Static terrain-conditioned experts, high-resolution local GFS and dynamic physical interactions.
        fine=16 if width<=24 else 32
        self.fuse=nn.Sequential(nn.Conv2d(width+45+24+6,fine,1),nn.GroupNorm(4,fine),nn.SiLU(),nn.Conv2d(fine,fine,3,padding=1,groups=4),nn.SiLU())
        self.thermo=nn.Sequential(nn.Conv2d(fine,fine,1),nn.SiLU(),nn.Conv2d(fine,2,1))
        self.wind=nn.Sequential(nn.Conv2d(fine,fine,1),nn.SiLU(),nn.Conv2d(fine,2,1))
    def up(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            return F.grid_sample(x.float(),self.grid.expand(len(x),-1,-1,-1),align_corners=True,padding_mode='border')
    def forward(self,x,t):
        n=len(x)
        if not self.physics:
            x=x.reshape(n,3,15,17,13); x=torch.cat([x[:,:,:4],torch.zeros_like(x[:,:,4:])],2).reshape(n,45,17,13)
        h=x.new_zeros(n,self.width,17,13); c=h
        for k in (2,1,0):
            z=self.stem(torch.cat([x[:,k*15:(k+1)*15],t[:,:,None,None].expand(-1,-1,17,13)],1))
            hn,cn=self.temporal(z,h,c); valid=t[:,8-k,None,None,None]; h=valid*hn+(1-valid)*h; c=valid*cn+(1-valid)*c
        low=self.context(h); seq=low.flatten(2).transpose(1,2)
        z=self.norm(seq+self.pos); a,_=self.attn(z,z,z,need_weights=False); ctx=(seq+a).transpose(1,2).reshape_as(low)
        raw=self.up(x); uv=raw[:,2:4]*self.sd[:,2:4]+self.mu[:,2:4]; u,v=uv[:,0:1],uv[:,1:2]
        sx,sy=self.geo[:,1:2],self.geo[:,2:3]; slope=torch.sqrt(sx*sx+sy*sy+1e-6)
        upslope=(u*sx+v*sy)/slope; along=(-u*sy+v*sx)/slope
        sh,ch,sd,cd=[t[:,i:i+1,None,None] for i in (0,1,4,5)]
        # Solar hour angle = UTC phase + longitude - pi; neglect equation of time.
        cosh=-(ch*torch.cos(self.longitude)-sh*torch.sin(self.longitude))
        sinh=-(sh*torch.cos(self.longitude)+ch*torch.sin(self.longitude))
        east=-cd*sinh; north=torch.cos(self.latitude)*sd-torch.sin(self.latitude)*cd*cosh
        up=torch.sin(self.latitude)*sd+torch.cos(self.latitude)*cd*cosh
        incidence=((up-sx*east-sy*north)/torch.sqrt(1+sx*sx+sy*sy)).clamp(0,1)*(up>0)
        dyn=torch.cat([upslope/5,along/5,(u*sx+v*sy)/2,up.clamp(0,1).expand(-1,-1,70,35),incidence,torch.sqrt(u*u+v*v+1e-6)/5],1)
        if not self.physics: dyn=torch.zeros_like(dyn)
        geo=self.geoenc(self.geo).expand(n,-1,-1,-1); h=self.fuse(torch.cat([self.up(ctx),raw,geo,dyn],1))
        return torch.cat([self.thermo(h),self.wind(h)],1).permute(0,2,3,1)

import hashlib, threading
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA
LOCK=threading.Lock(); LOG=None

def log(*args):
    with LOCK:
        msg=' '.join(map(str,args)); print(msg,flush=True)
        if LOG is not None:
            with LOG.open('a') as f: f.write(msg+'\n')

def load_all(root):
    root=Path(root); names=['train.npz','test.npz','static.npz','sample_submission.csv']
    if root.is_file():
        with zipfile.ZipFile(root) as z:
            data=[]
            for n in names[:3]:
                with np.load(io.BytesIO(z.read(n))) as f: data.append(dict(f))
            return (*data,pd.read_csv(io.BytesIO(z.read(names[3])),dtype={'id':str}))
    if not all((root/n).exists() for n in names):
        roots=[p.parent for p in root.rglob('train.npz') if all((p.parent/n).exists() for n in names)]
        if len(roots)!=1: raise ValueError(f'Dataset ambiguo: {roots}')
        root=roots[0]
    data=[]
    for n in names[:3]:
        with np.load(root/n) as f: data.append(dict(f))
    return (*data,pd.read_csv(root/names[3],dtype={'id':str}))

def anchor(X,t,h,Y,tr,ev,s,alpha=.08):
    """Train-only local ridge, atmospheric EOFs, and hourly anomalies."""
    start=time.perf_counter(); n=len(tr); H,W=Y.shape[1:3]
    a=X[:,:15]; b=(a-X[:,15:30])*t[:,7,None,None,None]; c=(a-X[:,30:45])*t[:,6,None,None,None]
    z=np.concatenate([a,b[:,:7],c[:,:7]],1).transpose(0,2,3,1).copy()
    z-=climate(z,h,tr)[h]; z/=np.maximum(z[tr].std((0,1,2)),1e-4); z=np.clip(z,-12,12)
    flat=z.reshape(len(z),-1); pca=PCA(n_components=min(24,n-1),svd_solver='randomized',random_state=17); pca.fit(flat[tr]); glob=pca.transform(flat); glob/=np.maximum(glob[tr].std(0),1e-4)
    yc=climate(Y,h,tr); target=((Y[tr]-yc[h[tr]])/NORM).reshape(n,-1,4)
    lat,lon=np.meshgrid(s['wrf_lats'],s['wrf_lons'],indexing='ij'); iy=np.abs(lat.ravel()[:,None]-s['gfs_lats']).argmin(1); ix=np.abs(lon.ravel()[:,None]-s['gfs_lons']).argmin(1)
    groups=iy*13+ix; both=np.r_[tr,ev]; pred=np.empty((len(both),H*W,4),dtype='f4')
    for g in np.unique(groups):
        r,c=divmod(int(g),13); pix=np.where(groups==g)[0]
        cells=sorted(set((int(np.clip(r+dr,0,16)),int(np.clip(c+dc,0,12))) for dr,dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]))
        d=np.concatenate([z[both,r,c],*[z[both,rr,cc,:7] for rr,cc in cells if (rr,cc)!=(r,c)],glob[both],t[both,:4],t[both,6:]],1).astype('f8')
        d=(d-d[:n].mean(0))/np.maximum(d[:n].std(0),.05); d=np.column_stack([np.ones(len(d)),d]); A=d[:n]; yy=target[:,pix].reshape(n,-1).astype('f8')
        gram=A.T@A; reg=np.full(len(gram),alpha*n); reg[0]=1e-6; gram.flat[::len(gram)+1]+=reg
        beta=cho_solve(cho_factor(gram),A.T@yy); pred[:,pix]=(d@beta).reshape(len(both),len(pix),4)
    pred=pred.reshape(-1,H,W,4)*NORM+yc[h[both]]; pred[...,0]=pred[...,0].clip(0,100)
    assert np.isfinite(pred).all(); log('Anchor',len(np.unique(groups)),'regions',round(time.perf_counter()-start,2),'s'); return pred[:n],pred[n:]

class AtlasResidual(TerrainFlow):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        for layer in [self.thermo[-1],self.wind[-1]]:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

def auxiliary(p,y,a):
    norm=p.new_tensor(NORM); p0=a+p*norm; y0=a+y*norm
    def vapor(x):
        c=(x[...,1]-273.15).clamp(-60,55); return 6.1094*torch.exp(17.625*c/(243.04+c))*x[...,0].clamp(0,100)/100
    speed=(torch.sqrt(p0[...,2:].square().sum(-1)+1e-5)-torch.sqrt(y0[...,2:].square().sum(-1)+1e-5))/2.5
    err=p-y
    return .02*speed.square().mean()+.015*((vapor(p0)-vapor(y0))/4).square().mean()+.01*(err.diff(dim=1).square().mean()+err.diff(dim=2).square().mean())

def fit(X,t,Y,tr,ev,at,ae,s,device,seed,epochs,kind,smoke,checkpoint=None):
    start=time.perf_counter(); mu=np.tile(X[tr,:15].mean((0,2,3)),3); sd=np.tile(np.maximum(X[tr,:15].std((0,2,3)),1e-4),3); geo,grid=geography(s)
    with LOCK:
        torch.manual_seed(seed); model=AtlasResidual(geo,grid,s['wrf_lats'],s['wrf_lons'],mu,sd,width=16 if smoke else 48,physics=kind!='RawResidual')
    model=model.to(device); ema=copy.deepcopy(model).eval(); ema.requires_grad_(False); rng=np.random.default_rng(seed); batch=16 if smoke else 32; amp=device.type=='cuda'
    xa=torch.as_tensor((X[tr]-mu[None,:,None,None])/sd[None,:,None,None],device=device); ta=torch.as_tensor(t[tr],device=device); aa=torch.as_tensor(at,device=device); ya=torch.as_tensor((Y[tr]-at)/NORM,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=math.ceil(len(tr)/batch)*epochs,eta_min=2e-5); scaler=torch.amp.GradScaler('cuda',enabled=amp); updates=0
    log(kind,device,seed,'params',sum(p.numel() for p in model.parameters()))
    for ep in range(epochs):
        model.train(); total=0.; perm=rng.permutation(len(tr))
        for off in range(0,len(tr),batch):
            ids=torch.as_tensor(perm[off:off+batch],device=device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): p=model(xa[ids],ta[ids])
            p=p.float(); mse=(p-ya[ids]).square().mean((0,1,2)); loss=.6*torch.sqrt(mse+1e-6).mean()+.4*mse.mean()
            if kind=='PhysicalResidual': loss=loss+auxiliary(p,ya[ids],aa[ids])
            if not torch.isfinite(loss): raise FloatingPointError('Loss no finita')
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.); prev=scaler.get_scale(); scaler.step(opt); scaler.update()
            if scaler.get_scale()>=prev:
                sched.step(); updates+=1
                with torch.no_grad(): torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-min(.995,(1+updates)/(10+updates)))
            total+=float(loss.detach())*len(ids)
        log(kind,device,seed,f'ep={ep+1}/{epochs} loss={total/len(tr):.6f} lr={opt.param_groups[0]["lr"]:.6f} elapsed={time.perf_counter()-start:.1f}s')
    preds=[]; ema.eval()
    with torch.inference_mode():
        for k in range(0,len(ev),batch):
            idx=ev[k:k+batch]; xb=torch.as_tensor((X[idx]-mu[None,:,None,None])/sd[None,:,None,None],device=device); tb=torch.as_tensor(t[idx],device=device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): p=ema(xb,tb)
            preds.append(p.float().cpu().numpy())
    pred=np.concatenate(preds)*NORM+ae; pred[...,0]=pred[...,0].clip(0,100); assert np.isfinite(pred).all()
    if checkpoint is not None: torch.save(dict(state_dict=ema.cpu().state_dict(),mu=mu,sd=sd,kind=kind,seed=seed),checkpoint)
    return pred

def ensemble(X,t,Y,tr,ev,at,ae,s,devices,epochs,kind,smoke,out=None):
    seeds=[42] if smoke else [42,123]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        fs=[pool.submit(fit,X,t,Y,tr,ev,at,ae,s,devices[i%len(devices)],seed,epochs,kind,smoke,None if out is None else out/f'{kind}_{seed}.pt') for i,seed in enumerate(seeds)]
        return np.mean([f.result() for f in fs],axis=0)

def weights(a,b,y):
    d=(b-a).astype('f8'); r=(y-a).astype('f8'); axes=(0,1,2); return (.85*np.clip((d*r).sum(axes)/np.maximum((d*d).sum(axes),1e-10),0,1)).astype('f4')

def blend(a,b,w):
    p=a+(b-a)*w; p[...,0]=p[...,0].clip(0,100); return p

def save_csv(template,times,p,dest):
    assert p.shape==(len(times),70,35,4)
    ids=[f'{t}_{r:02d}_{c:02d}' for t in times for r in range(70) for c in range(35)]
    assert len(ids)==646800 and len(template)==646800 and template.id.is_unique and set(ids)==set(template.id)
    d=pd.DataFrame(p.reshape(-1,4),columns=['RH','T','U','V'],index=ids).loc[template.id].reset_index(drop=True); d.insert(0,'id',template.id.to_numpy())
    assert np.isfinite(d.iloc[:,1:].to_numpy()).all() and d.RH.between(0,100).all(); tmp=dest.with_suffix('.tmp'); d.to_csv(tmp,index=False); tmp.replace(dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()

def main(root,out,smoke=False,anchor_only=False):
    global LOG
    start=time.perf_counter(); out=Path(out); out.mkdir(parents=True,exist_ok=True); LOG=out/'training.log'; LOG.write_text('')
    # Avoid mistaking an older successful run for a new one.
    for name in ('submission_atlas_v3_1.csv','report.json'):
        old=out/name
        if old.exists(): old.rename(out/(name+'.previous'))
    torch.set_num_threads(4 if smoke or anchor_only else 2); train,test,s,template=load_all(root); Y=train['Y']; T=train['times'].astype(str); TT=test['times'].astype(str)
    assert s['vars'].tolist()==['RH','T','U','V'] and np.all(T[1:]>T[:-1]) and np.all(TT[1:]>TT[:-1]) and T[-1]<TT[0]
    assert np.isfinite(Y).all() and np.isfinite(train['X']).all() and np.isfinite(test['X']).all()
    X,t,h=prepare(train['X'],T,s); XT,tT,hT=prepare(test['X'],TT,s)
    devices=[torch.device('cpu')] if smoke or anchor_only else [torch.device(f'cuda:{i}') for i in range(min(2,torch.cuda.device_count()))]
    if not devices: raise RuntimeError('Seleccionar GPU T4 x2')
    epochs=1 if smoke else 48; log('Atlas v3.1',devices,'smoke',smoke,'anchor_only',anchor_only,'epochs',epochs)
    rows=[]; models=['RawResidual','PhysicsResidual','PhysicalResidual']; chosen=None; learned=[]
    for i,(name,cut,cal,end) in enumerate(CUTS[:1] if smoke else CUTS):
        tr=np.where(T<cut)[0]; ca=np.where((T>=cut)&(T<cal)&(h%3==0))[0]; ev=np.where((T>=cal)&(T<end)&(h%3==0))[0]
        assert T[tr[-1]]<T[ca[0]] and T[ca[-1]]<T[ev[0]]
        if smoke: tr=tr[::13]
        both=np.r_[ca,ev]; at,ae=anchor(X,t,h,Y,tr,both,s); a=ae[len(ca):]; score,rm=metric(a,Y[ev]); rows.append(dict(fold=name,model='Anchor',skill=score,reference=BASE[name],**dict(zip(['RMSE_RH','RMSE_T','RMSE_U','RMSE_V'],rm.tolist())))); log(rows[-1])
        if not anchor_only:
            for kind in (models if i<2 else [chosen]):
                pred=ensemble(X,t,Y,tr,both,at,ae,s,devices,epochs,kind,smoke)
                w=weights(ae[:len(ca)],pred[:len(ca)],Y[ca]) if i<2 else np.mean([w for k,w in learned if k==kind],0); q=blend(a,pred[len(ca):],w); score,rm=metric(q,Y[ev]); rows.append(dict(fold=name,model=kind,skill=score,reference=BASE[name],weights=w.tolist(),**dict(zip(['RMSE_RH','RMSE_T','RMSE_U','RMSE_V'],rm.tolist())))); log(rows[-1]); np.savez_compressed(out/f'{name.replace(" ","_")}_{kind}.npz',times=T[ev],pred=q,anchor=a,neural=pred[len(ca):])
                if i<2: learned.append((kind,w))
            if i==1:
                chosen=max(models,key=lambda k:np.mean([r['skill'] for r in rows if r['model']==k])); log('FROZEN BEFORE AUDIT',chosen)
        pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False)
    report=dict(model='Atlas v3.1',smoke=smoke,anchor_only=anchor_only,results=rows,recommend_submit=False)
    if not anchor_only:
        if smoke: chosen=models[0]
        gains={r['fold']:r['skill']-r['reference'] for r in rows if r['model']==chosen}; gate=not smoke and len(gains)==4 and min(gains['Mayo'],gains['Junio I'])>0 and np.mean([gains['Mayo'],gains['Junio I']])>=.003 and gains['Junio II']>=.003 and gains['Abril']>=-.002
        report.update(chosen=chosen,gains=gains,recommend_submit=bool(gate)); log('VALIDATION_GATE',gate,'gains',gains)
        if gate or smoke:
            Xall=np.concatenate([X,XT]); tall=np.concatenate([t,tT]); hall=np.r_[h,hT]; tr=np.arange(len(T)); ev=len(T)+np.where(hT%3==0)[0]
            if smoke: tr=tr[::13]
            at,ae=anchor(Xall,tall,hall,Y,tr,ev,s); pred=ensemble(Xall,tall,Y,tr,ev,at,ae,s,devices,epochs,chosen,smoke,out); w=np.mean([w for k,w in learned if k==chosen],0); p=blend(ae,pred,w)
            dest=out/('SMOKE_NO_SUBIR.csv' if smoke else 'submission_atlas_v3_1.csv'); sha=save_csv(template,TT[hT%3==0],p,dest); report.update(weights=w.tolist(),sha256=sha,rows=len(template),csv=str(dest)); log(dest,sha)
        else: log('No CSV competitivo. Conservar Ultralisk v2.7.')
    report['seconds']=time.perf_counter()-start; (out/'report.json').write_text(json.dumps(report,indent=2)); log('RECOMMEND_SUBMIT='+('YES' if report['recommend_submit'] else 'NO'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data',default='/kaggle/input'); parser.add_argument('--out',default='/kaggle/working/Atlas-v3.1'); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--anchor-only',action='store_true'); args,_=parser.parse_known_args(); main(args.data,args.out,args.smoke,args.anchor_only)
