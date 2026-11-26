# Ultralisk-v2.7 | ShadowBytes | final calibration
import os, time, json, copy, hashlib, threading, platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from IPython.display import display, FileLink
SMOKE=os.environ.get('ULTRALISK_SMOKE')=='1'  # prueba tecnica, NO subir
OUT=Path(os.environ.get('ATLAS_OUT','./Atlas-v3.0-local')); OUT.mkdir(parents=True,exist_ok=True)
LOG=OUT/'training.log'; LOG.write_text(''); LOCK=threading.Lock(); START=time.perf_counter()
BATCH=32; EPOCHS_REF=25; EPOCHS_NEW=36
SEEDS={'CNN':(0,1,2),'ResNet':(0,1,2),'Mutalisk':(42,123),'Viper':(42,123)}
CUTS=[('2024050900','2024051600','2024060100'),('2024052500','2024060100','2024061600'),('2024060900','2024061600','2024070100')]
VARS=['RH','T','U','V']; NORM=np.array([25.360,2.0652,2.6776,2.0876])
# Valores historicos de versions/v3/backtest.json (solo referencia, no etiquetas).
HISTORICAL={1:.42292221906027727,2:.4136267261457105,3:.42723285641743414}
MIN_DEV_GAIN=.003; MIN_AUDIT_GAIN=.002
def log(msg):
    line=f'[{time.perf_counter()-START:8.1f}s] {msg}'
    with LOCK:
        print(line,flush=True)
        with LOG.open('a') as f: f.write(line+'\n')
DEVICES=[torch.device('cuda:0')] if torch.cuda.is_available() and not SMOKE else [torch.device('cpu')]
torch.set_num_threads(2); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
cpu_name=platform.processor()
if Path('/proc/cpuinfo').is_file():
    cpu_name=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),cpu_name)
log(f'torch={torch.__version__} CPU={cpu_name} cores={os.cpu_count()} torch_threads={torch.get_num_threads()} smoke={SMOKE}')
for d in DEVICES:
    if d.type=='cuda': log(f'{d} {torch.cuda.get_device_name(d)} VRAM={torch.cuda.get_device_properties(d).total_memory/2**30:.2f} GiB')
# Datos oficiales
root=Path(os.environ.get('ULTRALISK_DATA','/kaggle/input')); required=('train.npz','test.npz','static.npz','sample_submission.csv')
folders=[p.parent for p in root.rglob('train.npz') if all((p.parent/k).is_file() for k in required)]
official=[p for p in folders if 'hackathon-ia-pucp-2026' in str(p)]
if len(official)==1: DATA=official[0]
elif len(folders)==1: DATA=folders[0]
else: raise RuntimeError(f'Dataset oficial no identificado: {folders}')
with np.load(DATA/'train.npz') as f: X,Y,TT=f['X'].astype('f4'),f['Y'].astype('f4'),f['times'].astype(str)
with np.load(DATA/'test.npz') as f: XT,TE=f['X'].astype('f4'),f['times'].astype(str)
with np.load(DATA/'static.npz') as f:
    DEM=f['dem'].astype('f4'); glat,glon,wlat,wlon=[f[k] for k in ('gfs_lats','gfs_lons','wrf_lats','wrf_lons')]
TEMPLATE=pd.read_csv(DATA/'sample_submission.csv',dtype={'id':str})
assert X.shape==(2935,17,13,4) and Y.shape==(2935,70,35,4) and XT.shape==(792,17,13,4) and DEM.shape==(70,35)
assert TEMPLATE.shape==(646800,5) and TEMPLATE.columns.tolist()==['id',*VARS] and TEMPLATE.id.is_unique
assert all(np.isfinite(a).all() for a in (X,Y,XT,DEM)) and max(TT)<min(TE)
parts=TEMPLATE.id.str.split('_',expand=True); ids_t=parts[0].map({s:i for i,s in enumerate(TE)}); assert ids_t.notna().all()
ti=ids_t.to_numpy(dtype=int); ri=parts[1].astype(int).to_numpy(); ci=parts[2].astype(int).to_numpy()
assert (ri>=0).all() and (ri<70).all() and (ci>=0).all() and (ci<35).all()
assert len(np.unique(ti))==264 and np.all(np.bincount(ti)[np.unique(ti)]==2450)
HOURS=np.unique([int(s[-2:]) for s in parts[0].unique()]); H=np.array([int(s[-2:]) for s in TT]); HT=np.array([int(s[-2:]) for s in TE])
log(f'Data={DATA} train={X.shape} test={XT.shape} horas={HOURS.tolist()}')
def windows(x,times):
    dates=pd.to_datetime(times,format='%Y%m%d%H'); lookup={t:i for i,t in enumerate(dates)}
    return x[np.array([[lookup.get(t-pd.Timedelta(hours=k),i) for k in (2,1,0)] for i,t in enumerate(dates)])]
W,WT=windows(X,TT),windows(XT,TE)
lat,lon=np.meshgrid(wlat,wlon,indexing='ij'); grid=np.stack([(lon-glon[0])/(glon[-1]-glon[0])*2-1,(lat-glat[0])/(glat[-1]-glat[0])*2-1],-1)[None].astype('f4')
def z(a): return ((a-a.mean())/max(float(a.std()),1e-6)).astype('f4')
def smooth(a,k):
    p=k//2; return F.avg_pool2d(F.pad(torch.from_numpy(a[None,None]),(p,p,p,p),mode='replicate'),k,stride=1)[0,0].numpy()
dy,dx=np.gradient(DEM)
geo=np.stack([z(DEM),z(dx),z(dy),z(np.hypot(dx,dy)),z(DEM-smooth(DEM,3)),z(DEM-smooth(DEM,7)),z(DEM-smooth(DEM,15)),z(lat),z(lon),z(np.gradient(dy,axis=0)+np.gradient(dx,axis=1))])[None].astype('f4')
def stats(ix):
    yy,xx,hh=Y[ix],X[ix],H[ix]
    def climate(a): return np.stack([a[hh==h].mean(0) if np.any(hh==h) else a.mean(0) for h in range(24)])
    cl=climate(yy)
    return dict(mu=xx.mean((0,1,2)),sd=np.maximum(xx.std((0,1,2)),1e-6),clim=cl,gfs_clim=climate(xx),scale=np.maximum((yy-cl[hh]).std((0,1,2)),1e-6))
def score(p,y):
    rm=np.sqrt(np.mean(np.square(p.astype('f8')-y),axis=(0,1,2))); return float(1-(rm/NORM).mean()),rm
def clip(p):
    p=p.astype('f4',copy=False); p[...,0]=p[...,0].clip(0,100); assert np.isfinite(p).all(); return p
def tensor(a,d): return torch.as_tensor(a,dtype=torch.float32,device=d)
def sync(d):
    if d.type=='cuda': torch.cuda.synchronize(d)
def kernel_features(w,h,st):
    a=(w-st['gfs_clim'][(h[:,None]-np.array([2,1,0]))%24])/st['sd']; a=a.reshape(len(w),3,-1)
    return np.concatenate([a[:,2],a.mean(1),.5*(a[:,2]-a[:,0])],axis=1).astype('f4')
def kernel(a,b):
    dot=(a@b.T)/a.shape[1]; dist=(a.square().mean(1)[:,None]+b.square().mean(1)[None,:]-2*dot).clamp_min(0)
    return .5*dot+.25*dot.square()+.25*torch.exp(-dist/2)
CANDIDATES={'Reference':{},'Mutalisk':{'Mutalisk':1.},'Viper':{'Viper':1.},'Neural':{'Mutalisk':.5,'Viper':.5},'Kernel':{'Kernel':1.},'Hybrid':{'Mutalisk':.375,'Viper':.375,'Kernel':.25}}
def fit_alpha(a,b,y,prior=.5,shrink=.2,cap=1.):
    d=b.astype('f8')-a; r=y.astype('f8')-a; den=(d*d).mean((0,1,2)); num=(r*d).mean((0,1,2))
    raw=np.divide(num,den,out=np.full(4,prior),where=den>1e-12); return np.clip((1-shrink)*raw+shrink*prior,0,cap).astype('f4')
def mix(a,b,w): return clip(a*(1-w)+b*w)
def candidate(p,name): return sum(p[k]*np.float32(w) for k,w in CANDIDATES[name].items())
def calibrate(p,y,name):
    rw=fit_alpha(p['CNN'],p['ResNet'],y); ref=mix(p['CNN'],p['ResNet'],rw)
    alpha=np.zeros(4,dtype='f4') if name=='Reference' else fit_alpha(ref,candidate(p,name),y,prior=.25,shrink=.35,cap=.65)
    return rw,alpha
def combine(p,name,rw,alpha):
    ref=mix(p['CNN'],p['ResNet'],rw); return ref if name=='Reference' else mix(ref,candidate(p,name),alpha)


from scipy.optimize import minimize
PUBLIC_REFERENCE=.49056
OLD_FOLD={1:.439850572420165,2:.4396332592736687,3:.4496978946404211}
STRESS=('2024040900','2024041600','2024050100')
# Dos variantes conservadoras, probadas individualmente en CPU en folds 1 y 2.
CLOCK_SETTINGS=[(.6,90),(1.2,90)]
FAMILIES=['CNN','ResNet','Mutalisk','Viper','Kernel']
KEYS=FAMILIES+['Clock']
METHODS=['ClockReplace','ClockBlend','Stack']
MIN_DEV=.002; MIN_AUDIT=.002; STRESS_TOL=.002; REPRO_TOL=.0015

def old_weights(p,y):
    rw,alpha=calibrate(p,y,'Hybrid')
    # Convertir mezcla jerarquica v2.3 en pesos efectivos por variable.
    return np.stack([(1-alpha)*(1-rw),(1-alpha)*rw,.375*alpha,.375*alpha,.25*alpha,np.zeros(4,dtype='f4')])

def weighted(p,w):
    return clip(sum(p[k]*w[i].astype('f4') for i,k in enumerate(KEYS)))

def fit_method(p,y,method):
    w=old_weights(p,y)
    if method=='Baseline': return w
    if method=='ClockReplace':
        # Misma arquitectura y calibracion que v2.3, sustituyendo 75% del
        # componente kernel por la variante horaria/reciente.
        ref=mix(p['CNN'],p['ResNet'],fit_alpha(p['CNN'],p['ResNet'],y))
        fresh=.375*p['Mutalisk']+.375*p['Viper']+.0625*p['Kernel']+.1875*p['Clock']
        alpha=fit_alpha(ref,fresh,y,prior=.25,shrink=.35,cap=.65)
        rw=fit_alpha(p['CNN'],p['ResNet'],y)
        return np.stack([(1-alpha)*(1-rw),(1-alpha)*rw,.375*alpha,.375*alpha,.0625*alpha,.1875*alpha])
    if method=='ClockBlend':
        a=fit_alpha(weighted(p,w),p['Clock'],y,prior=.15,shrink=.4,cap=.55)
        w=w*(1-a); w[-1]+=a; return w
    if method!='Stack': raise ValueError(method)
    # Regresion convexa multisalida. Se ajusta solo en la semana de calibracion.
    # Penalizacion en escala de RMSE oficial y limites para evitar pesos extremos.
    out=[]
    for v in range(4):
        prior=w[:,v].astype('f8')*.85; prior[-1]+=.15
        # Residuos normalizados, matrices de solo 6x6.
        a=np.stack([((p[k][...,v]-y[...,v])/NORM[v]).reshape(-1) for k in KEYS],axis=0).astype('f8')
        gram=(a@a.T)/a.shape[1]; reg=.004
        def fun(z): return float(z@gram@z+reg*np.square(z-prior).sum())
        def jac(z): return 2*(gram@z+reg*(z-prior))
        bounds=[(max(0,float(t)-.25),min(1,float(t)+.25)) for t in prior]; bounds[-1]=(0,.55)
        opt=minimize(fun,prior,jac=jac,bounds=bounds,constraints=[dict(type='eq',fun=lambda z:z.sum()-1,jac=lambda z:np.ones(6))],method='SLSQP',options=dict(maxiter=100,ftol=1e-11))
        z=opt.x if opt.success and np.isfinite(opt.x).all() and abs(opt.x.sum()-1)<1e-6 else prior
        z=np.maximum(z,0); out.append(z/z.sum())
    return np.asarray(out,dtype='f4').T

def regional(d,st,tr,we,he,alpha=1.2,sigma=.65,local=.7,days=90,save_model=False,model_prefix="regional"):
    a=kernel_features(W[tr],H[tr],st); b=kernel_features(we,he,st); mu=a.mean(0); sd=np.maximum(a.std(0),.15)
    ag,bg=tensor((a-mu)/sd,d),tensor((b-mu)/sd,d)
    G=kernel(ag,ag); Q=kernel(bg,ag)
    h,eh=tensor(H[tr],d),tensor(he,d)
    clock=lambda a,b: .4+.6*(1+torch.cos((a[:,None]-b[None,:])*np.pi/12))/2
    C,D=clock(h,h),clock(eh,h)
    stamps=pd.to_datetime(TT[tr],format='%Y%m%d%H'); age=tensor(np.asarray((stamps.max()-stamps).total_seconds()/86400,dtype='f4')+1,d)
    yy=((Y[tr]-st['clim'][H[tr]])/st['scale']).reshape(len(tr),-1,4); pred=np.zeros((len(we),2450,4),np.float32); total=np.zeros(2450,np.float32)
    gy,gx=np.meshgrid(glat,glon,indexing='ij'); fy,fx=lat.ravel(),lon.ravel()
    if save_model: np.savez(OUT/f"{model_prefix}_preprocessing.npz",**st,features=(a-mu)/sd,feature_mu=mu,feature_sd=sd,train_hours=H[tr],train_times=TT[tr],alpha=alpha,sigma=sigma,local=local,days=days)
    anchor=0
    for cy in np.linspace(wlat[0],wlat[-1],3):
        for cx in np.linspace(wlon[0],wlon[-1],2):
            ww=np.exp(-.5*((gy-cy)**2+(gx-cx)**2)/sigma**2); ww=(ww/max(ww.mean(),1e-6))**.5; w=tensor(np.tile(np.repeat(ww.ravel(),4),3),d)
            k=((1-local)*G+local*kernel(ag*w,ag*w))*C; q=((1-local)*Q+local*kernel(bg*w,ag*w))*D
            k=(k+k.T)*.5; k.diagonal().add_(alpha*torch.exp(age/days)); L=torch.linalg.cholesky(k)
            blend=np.maximum(1-np.abs(fy-cy)/((wlat[-1]-wlat[0])/2),0)*np.maximum(1-np.abs(fx-cx)/(wlon[-1]-wlon[0]),0)
            ids=np.flatnonzero(blend>1e-6); target=tensor(yy[:,ids].reshape(len(tr),-1),d); dual=torch.cholesky_solve(target,L)
            if save_model: np.savez(OUT/f"{model_prefix}_sector_{anchor}.npz",indices=ids,blend=blend[ids],feature_weights=w.cpu().numpy(),dual=dual.cpu().numpy(),center=[cy,cx])
            anchor+=1
            pred[:,ids]+= (q@dual).cpu().numpy().reshape(len(we),len(ids),4)*blend[ids,None]; total[ids]+=blend[ids]
    assert total.min()>.99
    p=pred.reshape(len(we),70,35,4)/total.reshape(1,70,35,1)
    return clip(p*st['scale']+st['clim'][he])



def long_windows(x,times):
    dates=pd.to_datetime(times,format='%Y%m%d%H'); lookup={t:i for i,t in enumerate(dates)}
    return x[np.array([[lookup.get(t-pd.Timedelta(hours=k),i) for k in (2,1,0,6,12,24)] for i,t in enumerate(dates)])]
def history_features(w,h,st):
    lag=np.array([2,1,0,6,12,24]); a=(w-st['gfs_clim'][(h[:,None]-lag)%24])/st['sd']
    return np.stack([a[:,2],a[:,:3].mean(1),.5*(a[:,2]-a[:,0]),a[:,3],a[:,4],a[:,5]],axis=1).reshape(len(w),-1).astype('f4')
LW,LWT=long_windows(X,TT),long_windows(XT,TE)
def atlas_kernel(d,st,tr,we,he,alpha=1.2,sigma=.4,local=.85,days=90,wind=1.,save_model=False,model_prefix="atlas"):
    a=history_features(LW[tr],H[tr],st); b=history_features(we,he,st); mu=a.mean(0); sd=np.maximum(a.std(0),.15)
    channels=np.array([1,1,wind,wind],dtype="f4"); channels/=channels.mean()
    blocks=np.array([1,1,.7,.7,.5,.4],dtype="f4"); blocks/=blocks.mean()
    fw=np.sqrt(np.repeat(blocks,17*13*4)*np.tile(channels,6*17*13)).astype("f4")
    ag,bg=tensor((a-mu)/sd*fw,d),tensor((b-mu)/sd*fw,d)
    G=kernel(ag,ag); Q=kernel(bg,ag)
    h,eh=tensor(H[tr],d),tensor(he,d)
    clock=lambda a,b: .4+.6*(1+torch.cos((a[:,None]-b[None,:])*np.pi/12))/2
    C,D=clock(h,h),clock(eh,h)
    stamps=pd.to_datetime(TT[tr],format='%Y%m%d%H'); age=tensor(np.asarray((stamps.max()-stamps).total_seconds()/86400,dtype='f4')+1,d)
    yy=((Y[tr]-st['clim'][H[tr]])/st['scale']).reshape(len(tr),-1,4); pred=np.zeros((len(we),2450,4),np.float32); total=np.zeros(2450,np.float32)
    gy,gx=np.meshgrid(glat,glon,indexing='ij'); fy,fx=lat.ravel(),lon.ravel()
    if save_model: np.savez(OUT/f"{model_prefix}_preprocessing.npz",**st,features=(a-mu)/sd,feature_mu=mu,feature_sd=sd,train_hours=H[tr],train_times=TT[tr],alpha=alpha,sigma=sigma,local=local,days=days)
    anchor=0
    for cy in np.linspace(wlat[0],wlat[-1],3):
        for cx in np.linspace(wlon[0],wlon[-1],2):
            ww=np.exp(-.5*((gy-cy)**2+(gx-cx)**2)/sigma**2); ww=(ww/max(ww.mean(),1e-6))**.5; w=tensor(np.tile(np.repeat(ww.ravel(),4),6),d)
            k=((1-local)*G+local*kernel(ag*w,ag*w))*C; q=((1-local)*Q+local*kernel(bg*w,ag*w))*D
            k=(k+k.T)*.5; k.diagonal().add_(alpha*torch.exp(age/days)); L=torch.linalg.cholesky(k)
            blend=np.maximum(1-np.abs(fy-cy)/((wlat[-1]-wlat[0])/2),0)*np.maximum(1-np.abs(fx-cx)/(wlon[-1]-wlon[0]),0)
            ids=np.flatnonzero(blend>1e-6); target=tensor(yy[:,ids].reshape(len(tr),-1),d); dual=torch.cholesky_solve(target,L)
            if save_model: np.savez(OUT/f"{model_prefix}_sector_{anchor}.npz",indices=ids,blend=blend[ids],feature_weights=w.cpu().numpy(),dual=dual.cpu().numpy(),center=[cy,cx])
            anchor+=1
            pred[:,ids]+= (q@dual).cpu().numpy().reshape(len(we),len(ids),4)*blend[ids,None]; total[ids]+=blend[ids]
    assert total.min()>.99
    p=pred.reshape(len(we),70,35,4)/total.reshape(1,70,35,1)
    return clip(p*st['scale']+st['clim'][he])



from scipy.ndimage import gaussian_filter
rows=[]
def correct(p,y,h,he):
    e=y-p; glob=gaussian_filter(e.mean(0),sigma=(3,3,0)); maps=[]
    for hour in range(24):
        hw=np.exp(1.5*(np.cos((h-hour)*np.pi/12)-1)); local=np.average(e,axis=0,weights=hw)
        maps.append(.35*(.5*glob+.5*gaussian_filter(local,sigma=(3,3,0))))
    return np.asarray(maps,dtype='f4')[he]
for label,cut in [(1,CUTS[0]),(2,CUTS[1]),('March',('2024030900','2024031600','2024040100')),('MayEarly',('2024042400','2024050100','2024051600'))]:
    cal,start,end=cut; tr=np.flatnonzero(TT<cal); ca=np.flatnonzero((TT>=cal)&(TT<start)&np.isin(H,HOURS)); ev=np.flatnonzero((TT>=start)&(TT<end)&np.isin(H,HOURS)); ix=np.r_[ca,ev]; n=len(ca); st=stats(tr); preds={}
    for name in ['Narrow','History','WindHistory']:
        t=time.perf_counter(); p=regional(DEVICES[0],st,tr,W[ix],H[ix],alpha=1.2,sigma=.4,local=.85,days=90) if name=='Narrow' else atlas_kernel(DEVICES[0],st,tr,LW[ix],H[ix],wind=2. if name=='WindHistory' else 1.)
        preds[name]=p
        s,r=score(clip(p[n:]+correct(p[:n],Y[ca],H[ca],H[ev])),Y[ev]); row=dict(fold=label,model=name,skill=s,**dict(zip(['RMSE_'+v for v in VARS],r)),seconds=time.perf_counter()-t); rows.append(row); log(str(row)); pd.DataFrame(rows).to_csv(str(OUT/'validation_metrics.csv'),index=False)
    for name in ['History','WindHistory']:
        a,b=preds['Narrow'],preds[name]; w=fit_alpha(a[:n],b[:n],Y[ca],prior=.25,shrink=.5,cap=.5); p=mix(a,b,w); s,r=score(clip(p[n:]+correct(p[:n],Y[ca],H[ca],H[ev])),Y[ev]); row=dict(fold=label,model='Blend'+name,skill=s,**dict(zip(['RMSE_'+v for v in VARS],r)),seconds=0); rows.append(row); log(str(row)); pd.DataFrame(rows).to_csv(str(OUT/'validation_metrics.csv'),index=False)
