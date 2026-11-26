"""Atlas v3.0 PhysicalLSTM: independent atmospheric downscaling. Research only; no submission."""
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

BASE={'Mayo':.4599006,'Junio I':.4604175,'Junio II':.46508,'Abril':.55846}

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

def train_one(X,t,Y,h,tr,ev,s,device,seed,epochs,smoke,kind):
    mu=np.tile(X[tr,:15].mean((0,2,3)),3); sd=np.tile(np.maximum(X[tr,:15].std((0,2,3)),1e-4),3)
    clim=climate(Y,h,tr); geo,grid=geography(s)
    # Sequential RNG initialization in caller is not required here: use fork-free explicit seed before each worker via lock.
    with INIT_LOCK:
        torch.manual_seed(seed); model=TerrainFlow(geo,grid,s['wrf_lats'],s['wrf_lons'],mu,sd,width=16 if smoke else 48,physics=kind!='Raw')
    model=model.to(device); rng=np.random.default_rng(seed); batch=32
    def xpart(idx): return (X[idx]-mu[None,:,None,None])/sd[None,:,None,None]
    xa=torch.as_tensor(xpart(tr),device=device); ta=torch.as_tensor(t[tr],device=device)
    ya=torch.as_tensor((Y[tr]-clim[h[tr]])/NORM,device=device)
    xb=torch.as_tensor(xpart(ev),device=device); tb=torch.as_tensor(t[ev],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0005)
    steps=max(1,math.ceil(len(tr)/batch)*epochs); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=1e-5)
    amp=device.type=='cuda'; scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start=time.perf_counter()
    for ep in range(epochs):
        model.train(); total=0.; perm=rng.permutation(len(tr))
        for off in range(0,len(tr),batch):
            ids=torch.as_tensor(perm[off:off+batch],device=device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): p=model(xa[ids],ta[ids])
            err=p.float()-ya[ids]; mse=err.square().mean((0,1,2)); loss=.5*mse.mean()+.5*torch.sqrt(mse+1e-6).mean()
            if kind=="PhysicsReg": loss=loss+.02*(err.diff(dim=1).square().mean()+err.diff(dim=2).square().mean())
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            prev=scaler.get_scale(); scaler.step(opt); scaler.update()
            if scaler.get_scale()>=prev: sched.step()
            total+=float(loss.detach())*len(ids)
        if True:
            print(f'{kind} {device} seed={seed} epoch={ep+1}/{epochs} loss={total/len(tr):.6f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    model.eval(); preds=[]
    with torch.inference_mode():
        for k in range(0,len(ev),batch):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): p=model(xb[k:k+batch],tb[k:k+batch])
            preds.append(p.float().cpu().numpy())
    pred=np.concatenate(preds)*NORM+clim[h[ev]]; pred[...,0]=pred[...,0].clip(0,100)
    assert np.isfinite(pred).all(); return pred

import threading, builtins
INIT_LOCK=threading.Lock()
LOG_LOCK=threading.Lock()
LOG_PATH=None
def print(*args, **kwargs):
    with LOG_LOCK:
        builtins.print(*args, **kwargs)
        if LOG_PATH is not None:
            with LOG_PATH.open('a',encoding='utf-8') as stream:
                builtins.print(*args, file=stream, sep=kwargs.get('sep',' '),end=kwargs.get('end','\n'))
def main(root,out,pilot=False):
    global LOG_PATH
    start=time.perf_counter(); train,s=load(root); X,Y,T=train['X'],train['Y'],train['times'].astype(str); X,t,h=prepare(X,T,s)
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rows=[]; LOG_PATH=out/"training.log"; LOG_PATH.write_text("")
    torch.set_num_threads(4 if pilot else 2)
    devices=[torch.device('cpu')] if pilot else [torch.device(f'cuda:{i}') for i in range(min(2,torch.cuda.device_count()))]
    if not devices: raise RuntimeError('Select GPU T4 x2')
    epochs=int(os.environ.get('ATLAS_EPOCHS','3' if pilot else '40'))
    print('Atlas PhysicalLSTM |',devices,'| pilot=',pilot,'| features=',X.shape,'| epochs=',epochs,flush=True)
    for name,cut,cal,end in (CUTS[:int(os.environ.get('ATLAS_FOLDS','1'))] if pilot else CUTS):
        tr=np.where(T<cut)[0]; ev=np.where((T>=cal)&(T<end)&(h%3==0))[0]
        if pilot: tr=tr[h[tr]%3==0]
        seeds=[42] if pilot else [42,123]
        for kind in (os.environ.get('ATLAS_PILOT_MODELS','Raw,Physics').split(',') if pilot else ['Raw','Physics','PhysicsReg']):
            print(f'{name} {kind}: train={len(tr)} eval={len(ev)}',flush=True); now=time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures=[pool.submit(train_one,X,t,Y,h,tr,ev,s,devices[i%len(devices)],seed,epochs,pilot,kind) for i,seed in enumerate(seeds)]
                preds=[f.result() for f in futures]
            pred=np.mean(preds,axis=0); score,rm=metric(pred,Y[ev]); ref=BASE[name]
            row=dict(fold=name,model=kind,skill=score,reference=ref,gain=score-ref,pilot=pilot,epochs=epochs,train=len(tr),evaluation=len(ev),seconds=time.perf_counter()-now,**dict(zip(['RMSE_RH','RMSE_T','RMSE_U','RMSE_V'],rm.tolist())))
            rows.append(row); print(row,flush=True); pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False)
            np.savez_compressed(out/f'{name.replace(" ","_")}_{kind}.npz',times=T[ev],pred=pred)
    (out/'report.json').write_text(json.dumps(dict(pilot=pilot,seconds=time.perf_counter()-start,results=rows,research_only=True),indent=2))
    print(f'Finished in {time.perf_counter()-start:.2f}s | Research only: no submission.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='/kaggle/input'); p.add_argument('--out',default='/kaggle/working/Atlas-PhysicalLSTM'); p.add_argument('--pilot',action='store_true'); a,_=p.parse_known_args(); main(a.data,a.out,a.pilot)
