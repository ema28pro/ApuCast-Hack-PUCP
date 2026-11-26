# Ultralisk-v2.5 Atlas | ShadowBytes | cache extension for v2.4
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
OUT=Path(os.environ.get('ULTRALISK_ATLAS_OUT','/kaggle/working/Ultralisk-v2.5')); OUT.mkdir(parents=True,exist_ok=True)
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

def regional(d,st,tr,we,he,alpha=1.2,sigma=.65,local=.7,days=90,save_model=False):
    a=kernel_features(W[tr],H[tr],st); b=kernel_features(we,he,st); mu=a.mean(0); sd=np.maximum(a.std(0),.15)
    ag,bg=tensor((a-mu)/sd,d),tensor((b-mu)/sd,d)
    G=kernel(ag,ag); Q=kernel(bg,ag)
    h,eh=tensor(H[tr],d),tensor(he,d)
    clock=lambda a,b: .4+.6*(1+torch.cos((a[:,None]-b[None,:])*np.pi/12))/2
    C,D=clock(h,h),clock(eh,h)
    stamps=pd.to_datetime(TT[tr],format='%Y%m%d%H'); age=tensor(np.asarray((stamps.max()-stamps).total_seconds()/86400,dtype='f4')+1,d)
    yy=((Y[tr]-st['clim'][H[tr]])/st['scale']).reshape(len(tr),-1,4); pred=np.zeros((len(we),2450,4),np.float32); total=np.zeros(2450,np.float32)
    gy,gx=np.meshgrid(glat,glon,indexing='ij'); fy,fx=lat.ravel(),lon.ravel()
    if save_model: np.savez(OUT/"atlas_preprocessing.npz",**st,features=(a-mu)/sd,feature_mu=mu,feature_sd=sd,train_hours=H[tr],train_times=TT[tr],alpha=alpha,sigma=sigma,local=local,days=days)
    anchor=0
    for cy in np.linspace(wlat[0],wlat[-1],3):
        for cx in np.linspace(wlon[0],wlon[-1],2):
            ww=np.exp(-.5*((gy-cy)**2+(gx-cx)**2)/sigma**2); ww=(ww/max(ww.mean(),1e-6))**.5; w=tensor(np.tile(np.repeat(ww.ravel(),4),3),d)
            k=((1-local)*G+local*kernel(ag*w,ag*w))*C; q=((1-local)*Q+local*kernel(bg*w,ag*w))*D
            k=(k+k.T)*.5; k.diagonal().add_(alpha*torch.exp(age/days)); L=torch.linalg.cholesky(k)
            blend=np.maximum(1-np.abs(fy-cy)/((wlat[-1]-wlat[0])/2),0)*np.maximum(1-np.abs(fx-cx)/(wlon[-1]-wlon[0]),0)
            ids=np.flatnonzero(blend>1e-6); target=tensor(yy[:,ids].reshape(len(tr),-1),d); dual=torch.cholesky_solve(target,L)
            if save_model: np.savez(OUT/f"atlas_sector_{anchor}.npz",indices=ids,blend=blend[ids],feature_weights=w.cpu().numpy(),dual=dual.cpu().numpy(),center=[cy,cx])
            anchor+=1
            pred[:,ids]+= (q@dual).cpu().numpy().reshape(len(we),len(ids),4)*blend[ids,None]; total[ids]+=blend[ids]
    assert total.min()>.99
    p=pred.reshape(len(we),70,35,4)/total.reshape(1,70,35,1)
    return clip(p*st['scale']+st['clim'][he])


# ===================================================================
# ATLAS: usa las predicciones de cache del entrenamiento v2.4.
# No vuelve a entrenar las diez redes neuronales.
# ===================================================================
CACHE=Path(os.environ.get('ULTRALISK_CHRONOS_DIR','/kaggle/working/Ultralisk-v2.4'))
if not (CACHE/'report.json').is_file():
    raise RuntimeError(f'Primero ejecuta Ultralisk-v2.4 con ULTRALISK_CACHE=1. No existe {CACHE}/report.json')
report24=json.loads((CACHE/'report.json').read_text())
assert report24['model']=='Ultralisk-v2.4 Chronos', 'Cache incorrecta'
assert bool(report24['smoke'])==SMOKE, 'No se permite mezclar cache smoke con entrenamiento real'
ANCHOR=report24['selected'] if report24['recommend_submit'] and not SMOKE else 'Baseline'
assert ANCHOR in ['Baseline']+METHODS
log(f'Cache={CACHE} anchor={ANCHOR} smoke={SMOKE}')
for f in ['fold1_predictions.npz','fold2_predictions.npz','fold3_predictions.npz','foldApril_predictions.npz','final_family_predictions.npz']:
    if not (CACHE/f).is_file(): raise RuntimeError(f'Falta {CACHE/f}. Ejecuta v2.4 con ULTRALISK_CACHE=1.')
# Quitar de la salida cualquier candidato de una ejecucion anterior.
for filename in ['submission_ultralisk_v2_5.csv','submission_v2_5_NO_SUBIR.csv','submission_ATLAS_SMOKE_NO_SUBIR.csv']:
    (OUT/filename).unlink(missing_ok=True)
ATLAS_METHODS=['AtlasFixed20','AtlasAdaptive','AtlasReplace']
MIN_GAIN=.0015
BASELINES=['Baseline',ANCHOR]

def combine_atlas(p, y, method):
    # Ajusta pesos solo en el bloque de calibracion, nunca en el de evaluacion.
    w=fit_method(p,y,ANCHOR)
    if method=='Anchor': return np.vstack([w,np.zeros((1,4),dtype='f4')])
    if method=='AtlasReplace':
        # Reemplaza 75% de los kernels existentes, manteniendo todas las redes.
        a=.75*(w[4]+w[5]); w=w.copy(); w[4:6]*=.25
        return np.vstack([w,a])
    if method=='AtlasFixed20': a=np.full(4,.20,dtype='f4')
    elif method=='AtlasAdaptive':
        a=fit_alpha(weighted(p,w),p['Atlas'],y,prior=.20,shrink=.50,cap=.45)
    else: raise ValueError(method)
    return np.vstack([w*(1-a),a])

def atlas_predict(p,w):
    assert w.shape==(7,4) and np.all(w>=-1e-7) and np.allclose(w.sum(0),1,atol=1e-5)
    return clip(sum(p[k]*w[i].astype('f4') for i,k in enumerate(KEYS+['Atlas'])))

def read_cache(label,cut,methods):
    cal,start,end=cut
    tr=np.flatnonzero(TT<cal); ca=np.flatnonzero((TT>=cal)&(TT<start)&np.isin(H,HOURS)); ev=np.flatnonzero((TT>=start)&(TT<end)&np.isin(H,HOURS))
    if SMOKE: tr,ca,ev=tr[:48],ca[:4],ev[:4]
    ix=np.r_[ca,ev]; n=len(ca)
    assert len(tr) and len(ca) and len(ev) and max(TT[tr])<min(TT[ca])<min(TT[ev])
    with np.load(CACHE/f'fold{label}_predictions.npz',allow_pickle=False) as f:
        assert np.array_equal(f['times'],TT[ix]) and int(f['n_cal'])==n
        assert np.array_equal(f['target'],Y[ix]), 'Etiquetas de cache no coinciden con train.npz'
        p={k:f[k].astype('f4') for k in KEYS}
    assert all(v.shape==(len(ix),70,35,4) and np.isfinite(v).all() for v in p.values())
    t=time.perf_counter(); p['Atlas']=regional(DEVICES[0],stats(tr),tr,W[ix],H[ix])
    log(f'FOLD {label} Atlas train={len(tr)} cal={n} eval={len(ev)} seconds={time.perf_counter()-t:.2f}')
    pc={k:v[:n].copy() for k,v in p.items()}; pe={k:v[n:] for k,v in p.items()}; rows=[]; scores={}
    baseline=weighted(pe,fit_method(pc,Y[ca],'Baseline'))
    for method in ['V23','Anchor']+methods:
        if method=='V23': prediction=baseline
        else: prediction=atlas_predict(pe,combine_atlas(pc,Y[ca],method))
        sc,rm=score(prediction,Y[ev]); scores[method]=sc; rows.append(dict(fold=str(label),model=method,skill=sc,**dict(zip(['RMSE_'+v for v in VARS],rm.tolist())))); log(f'FOLD {label} {method} SKILL={sc:.6f} RMSE={rm.round(4).tolist()}')
    sc,rm=score(pe['Atlas'],Y[ev]); rows.append(dict(fold=str(label),model='AtlasIndividual',skill=sc,**dict(zip(['RMSE_'+v for v in VARS],rm.tolist())))); log(f'FOLD {label} AtlasIndividual SKILL={sc:.6f}')
    # Reproducir exactamente las metricas de v2.4 de la cache antes de seleccionar.
    historical=[r['skill'] for r in report24['validation'] if str(r['fold'])==str(label) and r['model']==ANCHOR]
    assert len(historical)==1 and abs(scores['Anchor']-historical[0])<2e-6, f'Cache no reproduce el anchor en {label}'
    historical23=[r['skill'] for r in report24['validation'] if str(r['fold'])==str(label) and r['model']=='Baseline']
    assert len(historical23)==1 and abs(scores['V23']-historical23[0])<2e-6
    return scores,rows,(pc,Y[ca])

ROWS=[]; DEV=[]; CAL=[]
for label,cut in enumerate(CUTS[:2],1):
    sc,rows,cal=read_cache(label,cut,ATLAS_METHODS); DEV.append(sc); ROWS+=rows; CAL.append(cal)
    pd.DataFrame(ROWS).to_csv(OUT/'validation_metrics.csv',index=False)
# Seleccion congelada antes de volver a evaluar junio III y abril.
gains={m:[s[m]-max(s['Anchor'],s['V23']) for s in DEV] for m in ATLAS_METHODS}
eligible=[m for m,g in gains.items() if min(g)>=0 and np.mean(g)>=MIN_GAIN]
selected=max(eligible,key=lambda m:np.mean(gains[m])) if eligible else 'Anchor'
log(f'FROZEN selected={selected} dev_gains={gains}')
for label,cut in [(3,CUTS[2]),('April',STRESS)]:
    sc,rows,cal=read_cache(label,cut,[selected] if selected!='Anchor' else []); ROWS+=rows
    if label==3: audit=sc; CAL.append(cal)
    else: stress=sc
    pd.DataFrame(ROWS).to_csv(OUT/'validation_metrics.csv',index=False)
reproduced=bool(report24['reference_reproduced']) and all(abs(s['V23']-OLD_FOLD[i])<=REPRO_TOL for i,s in enumerate(DEV+[audit],1))
audit_gain=audit[selected]-max(audit['Anchor'],audit['V23']); stress_gain=stress[selected]-max(stress['Anchor'],stress['V23'])
PASS=bool(not SMOKE and reproduced and selected!='Anchor' and audit_gain>=MIN_GAIN and stress_gain>=-.001)
log(f'GATE reproduced={reproduced} audit_gain={audit_gain:+.6f} stress_gain={stress_gain:+.6f} recommend={PASS}')
# Genera un CSV experimental aunque no pase: el nombre deja claro que NO debe subirse.
cal_p={k:np.concatenate([p[k] for p,y in CAL]) for k in KEYS+['Atlas']}; cal_y=np.concatenate([y for p,y in CAL]); weights=combine_atlas(cal_p,cal_y,selected)
log(f'FINAL weights={weights.tolist()}')
tr=np.argsort(TT); eval_t=np.unique(ti)
if SMOKE: tr=tr[:48]
with np.load(CACHE/'final_family_predictions.npz',allow_pickle=False) as f:
    assert np.array_equal(f['times'],TE[eval_t]); p={k:f[k].astype('f4') for k in KEYS}
assert all(v.shape==(len(eval_t),70,35,4) and np.isfinite(v).all() for v in p.values())
p['Atlas']=regional(DEVICES[0],stats(tr),tr,WT[eval_t],HT[eval_t],save_model=bool(PASS))
forecast=atlas_predict(p,weights)
positions=np.full(len(TE),-1,dtype=int); positions[eval_t]=np.arange(len(eval_t)); assert (positions[ti]>=0).all()
sub=pd.DataFrame(forecast[positions[ti],ri,ci].round(4),columns=VARS); sub.insert(0,'id',TEMPLATE.id)
name='submission_ATLAS_SMOKE_NO_SUBIR.csv' if SMOKE else ('submission_ultralisk_v2_5.csv' if PASS else 'submission_v2_5_NO_SUBIR.csv')
dest=OUT/name; tmp=dest.with_suffix('.tmp'); sub.to_csv(tmp,index=False)
check=pd.read_csv(tmp,dtype={'id':str}); assert check.shape==TEMPLATE.shape and check.columns.tolist()==TEMPLATE.columns.tolist() and check.id.equals(TEMPLATE.id) and check.id.is_unique
assert np.isfinite(check[VARS].to_numpy()).all() and check.RH.between(0,100).all(); tmp.replace(dest)
np.savez(OUT/'atlas_final_predictions.npz',times=TE[eval_t],Atlas=p['Atlas'])
report=dict(model='Ultralisk-v2.5 Atlas',smoke=SMOKE,anchor=ANCHOR,selected=selected,reference_public_score=.49056,reference_reproduced=reproduced,dev_gains=gains,audit_gain=audit_gain,stress_gain=stress_gain,recommend_submit=PASS,keys=KEYS+['Atlas'],weights=weights.tolist(),validation=ROWS,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),rows=len(check),seconds=time.perf_counter()-START)
(OUT/'report.json').write_text(json.dumps(report,indent=2)); print(pd.DataFrame(ROWS).pivot(index='model',columns='fold',values='skill').round(6).to_string())
log(f'CSV={dest} rows={len(check)} SHA256={report["sha256"]}'); log('RECOMMEND_SUBMIT=YES' if PASS else 'RECOMMEND_SUBMIT=NO: conservar mejor CSV ya validado')
for f in (dest,OUT/'report.json',OUT/'validation_metrics.csv',LOG): display(FileLink(os.path.relpath(f)))
