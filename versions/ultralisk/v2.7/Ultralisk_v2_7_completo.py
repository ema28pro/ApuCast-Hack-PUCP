# ULTRALISK v2.7 COMPLETO | ShadowBytes | una sola celda Kaggle
import os, json
from pathlib import Path
_cache27 = Path(os.environ.get('ULTRALISK_CHRONOS_DIR','/kaggle/working/Ultralisk-v2.4'))
_cachefiles27 = ['report.json','fold1_predictions.npz','fold2_predictions.npz','fold3_predictions.npz','foldApril_predictions.npz','final_family_predictions.npz']
_ready27 = all((_cache27/f).is_file() for f in _cachefiles27)
if _ready27:
    try:
        _r27 = json.loads((_cache27/'report.json').read_text())
        _ready27 = _r27.get('model')=='Ultralisk-v2.4 Chronos' and bool(_r27.get('smoke'))==(os.environ.get('ULTRALISK_SMOKE')=='1')
    except (OSError, ValueError): _ready27=False
if not _ready27:
    print('FASE 1/2: creando cache neuronal v2.4. NO subir sus CSV intermedios.',flush=True)
    os.environ['ULTRALISK_OUT']=str(_cache27)
    os.environ['ULTRALISK_CACHE']='1'
    # Ultralisk-v2.4 | ShadowBytes | ApuCast | Kaggle T4 x2
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
    OUT=Path(os.environ.get('ULTRALISK_OUT','/kaggle/working/Ultralisk-v2.4')); OUT.mkdir(parents=True,exist_ok=True)
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
    if not SMOKE and (torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2))): raise RuntimeError('Selecciona Accelerator > GPU T4 x2')
    DEVICES=[torch.device('cpu')] if SMOKE else [torch.device('cuda:0'),torch.device('cuda:1')]
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
    # Arquitecturas
    class Residual(nn.Module):
        def __init__(self,a,b,dilation=1):
            super().__init__(); self.c1=nn.Conv2d(a,b,3,padding=dilation,dilation=dilation); self.c2=nn.Conv2d(b,b,3,padding=1); self.skip=nn.Conv2d(a,b,1) if a!=b else nn.Identity()
        def forward(self,x): return F.gelu(self.c2(F.gelu(self.c1(x)))+self.skip(x))
    def inputs(x,h):
        n=len(x); a=h*(2*np.pi/24); clock=torch.stack([a.sin(),a.cos()],1)[:,:,None,None].expand(-1,-1,17,13)
        return x.permute(0,1,4,2,3).reshape(n,12,17,13),clock
    class Reference(nn.Module):
        def __init__(self,kind):
            super().__init__(); w=64
            self.enc=nn.Sequential(nn.Conv2d(14,w,3,padding=1),nn.GELU(),nn.Conv2d(w,w,3,padding=1),nn.GELU())
            self.static=nn.Parameter(torch.zeros(1,8,70,35))
            self.dec=nn.Sequential(nn.Conv2d(73,w,3,padding=1),nn.GELU(),nn.Conv2d(w,w,3,padding=1),nn.GELU(),nn.Conv2d(w,4,1))
            if kind=='ResNet': self.enc=Residual(14,w); self.dec=nn.Sequential(Residual(73,w),nn.Conv2d(w,4,1))
            self.register_buffer('grid',torch.from_numpy(grid)); self.register_buffer('dem',torch.from_numpy(z(DEM)[None,None]))
        def forward(self,x,h):
            n=len(x); fields,clock=inputs(x,h); v=self.enc(torch.cat([fields,clock],1))
            with torch.autocast(device_type=x.device.type,enabled=False): v=F.grid_sample(v.float(),self.grid.expand(n,-1,-1,-1),align_corners=True)
            return self.dec(torch.cat([v,self.dem.expand(n,-1,-1,-1),self.static.expand(n,-1,-1,-1)],1)).permute(0,2,3,1)
    class GeoWind(nn.Module):
        def __init__(self,kind,st):
            super().__init__(); self.kind=kind; w=48
            self.register_buffer('grid',torch.from_numpy(grid)); self.register_buffer('geo',torch.from_numpy(geo)); self.register_buffer('gcl',torch.from_numpy(((st['gfs_clim']-st['mu'])/st['sd']).astype('f4')))
            self.enc=nn.Sequential(Residual(22,w),Residual(w,w,2),Residual(w,w)); self.map=nn.Parameter(torch.zeros(1,8,70,35)); self.fuse=Residual(w+26,w)
            if kind=='Mutalisk': self.down=nn.Sequential(Residual(w,64),Residual(64,64,2)); self.up=Residual(w+64,w)
            else: self.wind=nn.Sequential(Residual(w,w,2),Residual(w,w,3))
            self.thermal=nn.Sequential(Residual(w,32),nn.Conv2d(32,2,1)); self.velocity=nn.Sequential(Residual(w,32),nn.Conv2d(32,2,1))
        def interp(self,x):
            with torch.autocast(device_type=x.device.type,enabled=False): return F.grid_sample(x.float(),self.grid.expand(len(x),-1,-1,-1),align_corners=True,padding_mode='border')
        def forward(self,x,h):
            n=len(x); fields,clock=inputs(x,h); current=x[:,-1]; anomaly=current-self.gcl[h.long()]; trend=current-x[:,0]
            av=anomaly.permute(0,3,1,2); cv=current.permute(0,3,1,2); v=self.enc(torch.cat([fields,av,trend.permute(0,3,1,2),clock],1))
            fine=self.fuse(torch.cat([self.interp(v),self.interp(torch.cat([cv,av],1)),self.geo.expand(n,-1,-1,-1),self.map.expand(n,-1,-1,-1)],1))
            if self.kind=='Mutalisk':
                low=self.down(F.avg_pool2d(fine,2)); fine=self.up(torch.cat([fine,F.interpolate(low,size=fine.shape[-2:],mode='bilinear',align_corners=False)],1)); wind=fine
            else: wind=self.wind(fine)
            return torch.cat([self.thermal(fine),self.velocity(wind)],1).permute(0,2,3,1)
    def make(kind,st): return Reference(kind) if kind in ('CNN','ResNet') else GeoWind(kind,st)
    # Benchmark de inferencia, mismo modelo y lote para todos los dispositivos.
    def benchmark():
        st=stats(np.arange(128)); torch.manual_seed(0); base=Reference('ResNet').eval(); results=[]
        x=(W[:8]-st['mu'])/st['sd']; h=H[:8]
        for d,amp in [(torch.device('cpu'),False)]+[(d,a) for d in DEVICES if d.type=='cuda' for a in (False,True)]:
            m=copy.deepcopy(base).to(d,memory_format=torch.channels_last); xb,hb=tensor(x,d),tensor(h,d)
            with torch.inference_mode():
                for _ in range(3):
                    with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=amp): m(xb,hb)
                sync(d); t=time.perf_counter()
                for _ in range(8):
                    with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=amp): m(xb,hb)
                sync(d); sec=time.perf_counter()-t
            r=dict(device=str(d),amp=amp,batch=8,ms_per_batch=sec/8*1000,samples_per_second=64/sec); results.append(r); log(f'BENCHMARK {r}')
        if len(DEVICES)==2:
            barrier=threading.Barrier(2)
            def both(d):
                m=copy.deepcopy(base).to(d,memory_format=torch.channels_last); xb,hb=tensor(x,d),tensor(h,d)
                with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.float16):
                    for _ in range(3): m(xb,hb)
                    sync(d); barrier.wait(timeout=60); t=time.perf_counter()
                    for _ in range(8): m(xb,hb)
                    sync(d); return time.perf_counter()-t
            with ThreadPoolExecutor(max_workers=2) as pool:
                sec=max(pool.map(both,DEVICES))
            r=dict(device='T4x2_parallel',amp=True,batch=16,ms_per_batch=sec/8*1000,samples_per_second=128/sec)
            results.append(r); log(f'BENCHMARK {r}')
        return results
    BENCH=benchmark() if os.environ.get('ULTRALISK_BENCH','0')=='1' else []
    log('Benchmark opcional; iniciando experimentos v2.4')
    @torch.inference_mode()
    def infer(model,x,h,d,amp):
        model.eval(); out=[]
        for i in range(0,len(x),BATCH):
            with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=amp): p=model(x[i:i+BATCH],h[i:i+BATCH])
            out.append(p.float().cpu().numpy())
        return np.concatenate(out)
    def fit_neural(job,d,st,tr,we,he,tag):
        kind,seed,model=job; ref=kind in ('CNN','ResNet'); epochs=1 if SMOKE else (EPOCHS_REF if ref else EPOCHS_NEW)
        amp=not ref and d.type=='cuda'; lr=.002 if ref else .0012; rng=np.random.RandomState(seed)
        model=model.to(d,memory_format=torch.channels_last); ema=None if ref else copy.deepcopy(model).eval().requires_grad_(False)
        x=tensor((W[tr]-st['mu'])/st['sd'],d); h=tensor(H[tr],d); y=tensor((Y[tr]-st['clim'][H[tr]])/st['scale'],d)
        xe=tensor((we-st['mu'])/st['sd'],d); he_gpu=tensor(he,d)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4 if ref else 5e-4)
        steps=(len(tr)+BATCH-1)//BATCH; sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps*epochs)
        scaler=torch.amp.GradScaler('cuda',enabled=amp); ratio=tensor(st['scale']/NORM,d)
        order=np.arange(len(tr)); started=time.perf_counter(); hist=[]
        if d.type=='cuda': torch.cuda.reset_peak_memory_stats(d)
        log(f'{tag} {kind}:{seed} {d} START params={sum(p.numel() for p in model.parameters()):,} epochs={epochs} amp={amp}')
        for ep in range(1,epochs+1):
            model.train(); rng.shuffle(order); accum=torch.zeros((),device=d); t0=time.perf_counter()
            for k in range(0,len(order),BATCH):
                ix=torch.as_tensor(order[k:k+BATCH],device=d,dtype=torch.long); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=amp): p=model(x[ix],h[ix])
                mse=(p.float()-y[ix]).square(); loss=mse.mean() if ref else .5*mse.mean()+.5*((mse.mean((0,1,2))+1e-8).sqrt()*ratio).mean()
                scaler.scale(loss).backward()
                if not ref: scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.)
                old=scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale()>=old:
                    sched.step()
                    if ema is not None:
                        with torch.no_grad():
                            for a,b in zip(ema.parameters(),model.parameters()): a.lerp_(b,.015)
                accum+=loss.detach()*len(ix)
            sync(d); sec=time.perf_counter()-t0; value=float(accum.item()/len(tr)); assert np.isfinite(value)
            mem=torch.cuda.max_memory_allocated(d)/2**30 if d.type=='cuda' else None
            hist.append(dict(epoch=ep,loss=value,seconds=sec,samples_per_second=len(tr)/sec,peak_gib=mem,lr=opt.param_groups[0]['lr']))
            log(f'{tag} {kind}:{seed} {d} ep={ep:02d}/{epochs} loss={value:.5f} lr={opt.param_groups[0]["lr"]:.6f} {sec:.2f}s {len(tr)/sec:.1f} samples/s VRAM={mem}')
        chosen=model if ref else ema; p=clip(infer(chosen,xe,he_gpu,d,amp)*st['scale']+st['clim'][he]); elapsed=time.perf_counter()-started
        pd.DataFrame(hist).to_csv(OUT/f'{tag}_{kind}_{seed}_history.csv',index=False)
        if tag=='final': torch.save({'kind':kind,'seed':seed,'epochs':epochs,'state_dict':{k:v.detach().cpu() for k,v in chosen.state_dict().items()}},OUT/f'{kind}_{seed}.pt')
        log(f'{tag} {kind}:{seed} DONE {elapsed:.1f}s'); model.to('cpu')
        return kind,p,dict(kind=kind,seed=seed,device=str(d),seconds=elapsed,epochs=epochs,peak_gib=mem)
    # Kernel Ridge no lineal: anomalias del GFS, tendencia y promedio temporal.
    def kernel_features(w,h,st):
        a=(w-st['gfs_clim'][(h[:,None]-np.array([2,1,0]))%24])/st['sd']; a=a.reshape(len(w),3,-1)
        return np.concatenate([a[:,2],a.mean(1),.5*(a[:,2]-a[:,0])],axis=1).astype('f4')
    def kernel(a,b):
        dot=(a@b.T)/a.shape[1]; dist=(a.square().mean(1)[:,None]+b.square().mean(1)[None,:]-2*dot).clamp_min(0)
        return .5*dot+.25*dot.square()+.25*torch.exp(-dist/2)
    def fit_kernel(d,st,tr,we,he,tag):
        t=time.perf_counter(); a=kernel_features(W[tr],H[tr],st); b=kernel_features(we,he,st)
        mu=a.mean(0); sd=np.maximum(a.std(0),.15); a=(a-mu)/sd; b=(b-mu)/sd
        ag,bg=tensor(a,d),tensor(b,d); K=kernel(ag,ag); K=(K+K.T)*.5; K.diagonal().add_(.05)
        L=torch.linalg.cholesky(K); yy=tensor(((Y[tr]-st['clim'][H[tr]])/st['scale']).reshape(len(tr),-1),d)
        dual=torch.cholesky_solve(yy,L); p=(kernel(bg,ag)@dual).cpu().numpy().reshape(len(we),70,35,4)
        if tag=='final': np.savez_compressed(OUT/'kernel_model.npz',features=a,mu=mu,sd=sd,dual=dual.cpu().numpy())
        log(f'{tag} Kernel DONE {time.perf_counter()-t:.2f}s alpha=.05'); return clip(p*st['scale']+st['clim'][he])
    ALL_RUNS=[]
    def run_stage(tr,we,he,tag,families):
        st=stats(tr); jobs=[]
        for kind in families:
            if kind=='Kernel': continue
            for seed in (SEEDS[kind][:1] if SMOKE else SEEDS[kind]):
                torch.manual_seed(seed); jobs.append((kind,seed,make(kind,st)))
        def worker(group,d): return [fit_neural(j,d,st,tr,we,he,tag) for j in group]
        results=[]
        if len(DEVICES)==2:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker,jobs[i::2],d) for i,d in enumerate(DEVICES)]
                for f in futures: results.extend(f.result())
        else: results=worker(jobs,DEVICES[0])
        ALL_RUNS.extend([dict(tag=tag,**r) for _,_,r in results]); results.sort(key=lambda x:x[2]['seed'])
        pred={k:np.mean([p for kind,p,_ in results if kind==k],axis=0,dtype=np.float32) for k in families if k!='Kernel'}
        if 'Kernel' in families: pred['Kernel']=fit_kernel(DEVICES[0],st,tr,we,he,tag)
        if 'Kernel' in families: pred['Clock']=fit_clock(DEVICES[0],st,tr,we,he,tag)
        if tag=='final': np.savez_compressed(OUT/'preprocessing.npz',**st,grid=grid,geo=geo,normalizers=NORM)
        return pred
    # Calibracion por variable, separada de evaluacion.
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


    # ================================================================
    # ULTRALISK v2.4 | CHRONOS | extension incremental de v2.3
    # No usa WRF oculto, datos externos ni horas futuras de GFS.
    # ================================================================
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

    def fit_clock(d,st,tr,we,he,tag):
        t=time.perf_counter()
        a=kernel_features(W[tr],H[tr],st); b=kernel_features(we,he,st)
        mu=a.mean(0); sd=np.maximum(a.std(0),.15)
        a=(a-mu)/sd; b=(b-mu)/sd
        ag,bg=tensor(a,d),tensor(b,d)
        h,eh=tensor(H[tr],d),tensor(he,d)
        def clock(a,b): return .4+.6*(1+torch.cos((a[:,None]-b[None,:])*(np.pi/12)))/2
        K=kernel(ag,ag)*clock(h,h); Q=kernel(bg,ag)*clock(eh,h)
        K=(K+K.T)*.5
        stamps=pd.to_datetime(TT[tr],format='%Y%m%d%H')
        age=tensor(np.asarray((stamps.max()-stamps).total_seconds()/86400,dtype='f4'),d)
        # Edad calculada solo desde fechas del entrenamiento. Un dia mas aproxima
        # el comienzo del siguiente periodo de calibracion (igual en todos los folds).
        age=age+1.
        target=tensor(((Y[tr]-st['clim'][H[tr]])/st['scale']).reshape(len(tr),-1),d)
        preds=[]
        for alpha,days in CLOCK_SETTINGS:
            Kreg=K.clone(); Kreg.diagonal().add_(alpha*torch.exp(age/days))
            L=torch.linalg.cholesky(Kreg); dual=torch.cholesky_solve(target,L)
            p=(Q@dual).cpu().numpy().reshape(len(we),70,35,4)
            preds.append(clip(p*st['scale']+st['clim'][he]))
            if tag=='final':
                np.savez_compressed(OUT/f'clock_alpha_{alpha}.npz',features=a,mu=mu,sd=sd,dual=dual.cpu().numpy(),hours=H[tr],alpha=alpha,days=days)
            del Kreg,L,dual,p
            log(f'{tag} Clock alpha={alpha} days={days} done elapsed={time.perf_counter()-t:.2f}s')
        return clip(np.mean(preds,axis=0,dtype=np.float32))

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

    # Cache de predicciones de calibracion y validacion para futuras versiones.
    # Las etiquetas se obtienen exclusivamente de train.npz.
    def fold_run(label,cut,methods):
        cal,start,end=cut
        tr=np.flatnonzero(TT<cal); ca=np.flatnonzero((TT>=cal)&(TT<start)&np.isin(H,HOURS)); ev=np.flatnonzero((TT>=start)&(TT<end)&np.isin(H,HOURS))
        if SMOKE: tr,ca,ev=tr[:48],ca[:4],ev[:4]
        assert len(tr) and len(ca) and len(ev) and max(TT[tr])<min(TT[ca])<min(TT[ev])
        log(f'FOLD {label} train={len(tr)} cal={len(ca)} eval={len(ev)} {cal}/{start}/{end}')
        ix=np.r_[ca,ev]; p=run_stage(tr,W[ix],H[ix],f'fold{label}',FAMILIES)
        pc={k:v[:len(ca)].copy() for k,v in p.items()}; pe={k:v[len(ca):] for k,v in p.items()}
        scores={}; rows=[]
        for method in methods:
            weights=fit_method(pc,Y[ca],method); pred=weighted(pe,weights); s,r=score(pred,Y[ev]); scores[method]=s
            rows.append(dict(fold=str(label),model=method,skill=s,**dict(zip(['RMSE_'+v for v in VARS],r.tolist()))))
            log(f'FOLD {label} {method:13s} skill={s:.6f} RMSE={np.round(r,4).tolist()} weights={weights.round(3).tolist()}')
        if os.environ.get('ULTRALISK_CACHE','1')=='1':
            np.savez(OUT/f'fold{label}_predictions.npz',times=TT[ix],n_cal=len(ca),target=Y[ix],**p)
        return scores,rows,(pc,Y[ca])

    ROWS=[]; CAL=[]; DEV=[]
    for f,cut in enumerate(CUTS[:2],1):
        sc,rows,cal=fold_run(f,cut,['Baseline']+METHODS); DEV.append(sc); ROWS+=rows; CAL.append(cal)
        pd.DataFrame(ROWS).to_csv(OUT/'validation_metrics.csv',index=False)
    dev_gains={m:[s[m]-s['Baseline'] for s in DEV] for m in METHODS}
    repro=all(abs(s['Baseline']-OLD_FOLD[i])<=REPRO_TOL for i,s in enumerate(DEV,1))
    eligible=[m for m,g in dev_gains.items() if min(g)>=0 and np.mean(g)>=MIN_DEV]
    selected=max(eligible,key=lambda m:np.mean(dev_gains[m])) if eligible else 'Baseline'
    log(f'FROZEN BEFORE CHECKS selected={selected} gains={dev_gains} baseline_reproduced={repro}')
    methods=list(dict.fromkeys(['Baseline',selected]))
    # Junio III ya fue observado durante el desarrollo de v2.3: regresion historica,
    # no un nuevo test ciego. Abril es un corte adicional de estres cronologico.
    audit,rows,cal=fold_run(3,CUTS[2],methods); ROWS+=rows; CAL.append(cal)
    pd.DataFrame(ROWS).to_csv(OUT/'validation_metrics.csv',index=False)
    repro=repro and abs(audit['Baseline']-OLD_FOLD[3])<=REPRO_TOL
    stress,rows,_=fold_run('April',STRESS,methods); ROWS+=rows
    pd.DataFrame(ROWS).to_csv(OUT/'validation_metrics.csv',index=False)
    gain=audit[selected]-audit['Baseline']; stress_gain=stress[selected]-stress['Baseline']
    PASS=selected!='Baseline' and repro and gain>=MIN_AUDIT and stress_gain>=-STRESS_TOL and not SMOKE
    log(f'GATE selected={selected} june_gain={gain:+.6f} april_gain={stress_gain:+.6f} reproduced={repro} recommend={PASS}')
    # La calibracion final concatena solo las tres semanas originales, no evaluaciones.
    # No se usa el puntaje publico como etiqueta ni para optimizar pesos.
    cal_p={k:np.concatenate([p[k] for p,y in CAL]) for k in KEYS}; cal_y=np.concatenate([y for p,y in CAL]); weights=fit_method(cal_p,cal_y,selected)
    log(f'FINAL weights keys={KEYS} weights={weights.tolist()}')
    tr=np.argsort(TT); eval_t=np.unique(ti)
    if SMOKE: tr=tr[:48]
    p=run_stage(tr,WT[eval_t],HT[eval_t],'final',FAMILIES)
    forecast=weighted(p,weights)
    positions=np.full(len(TE),-1,dtype=int); positions[eval_t]=np.arange(len(eval_t))
    values=forecast[positions[ti],ri,ci]; sub=pd.DataFrame(values.round(4),columns=VARS); sub.insert(0,'id',TEMPLATE.id)
    name='submission_ultralisk_v2_4.csv' if PASS else 'submission_v2_4_NO_SUBIR.csv'
    if SMOKE: name='submission_SMOKE_NO_SUBIR.csv'
    dest=OUT/name; tmp=dest.with_suffix('.tmp'); sub.to_csv(tmp,index=False)
    check=pd.read_csv(tmp,dtype={'id':str}); assert check.shape==TEMPLATE.shape and check.columns.tolist()==TEMPLATE.columns.tolist() and check.id.equals(TEMPLATE.id) and check.id.is_unique
    assert np.isfinite(check[VARS].to_numpy()).all() and check.RH.between(0,100).all(); tmp.replace(dest)
    if os.environ.get('ULTRALISK_CACHE','1')=='1': np.savez(OUT/'final_family_predictions.npz',times=TE[eval_t],**p)
    report=dict(model='Ultralisk-v2.4 Chronos',smoke=SMOKE,selected=selected,reference_public_score=PUBLIC_REFERENCE,reference_reproduced=repro,dev_gains=dev_gains,audit_gain=gain,stress_gain=stress_gain,recommend_submit=bool(PASS),keys=KEYS,weights=weights.tolist(),clock_settings=CLOCK_SETTINGS,validation=ROWS,benchmark=BENCH,runs=ALL_RUNS,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),rows=len(check),seconds=time.perf_counter()-START)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(pd.DataFrame(ROWS).pivot(index='model',columns='fold',values='skill').round(6).to_string())
    log(f'CSV={dest} SHA256={report["sha256"]} rows={len(check)}')
    log('RECOMMEND_SUBMIT=YES' if PASS else 'RECOMMEND_SUBMIT=NO: conservar CSV v2.3 (0.49056)')
    for f in (dest,OUT/'report.json',OUT/'validation_metrics.csv',LOG): display(FileLink(os.path.relpath(f)))
else:
    print('FASE 1/2: cache v2.4 existente; no se reentrenan redes.',flush=True)
print('FASE 2/2: calibracion final Ultralisk-v2.7',flush=True)

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
OUT=Path(os.environ.get('ULTRALISK_V27_OUT','/kaggle/working/Ultralisk-v2.7')); OUT.mkdir(parents=True,exist_ok=True)
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


# ===================================================================
# Usa las predicciones de cache del entrenamiento v2.4.
# No vuelve a entrenar las diez redes neuronales.
# ===================================================================
# ULTRALISK 2.7 - calibracion final. No se realizan submissions.
from scipy.ndimage import gaussian_filter
CACHE=Path(os.environ.get('ULTRALISK_CHRONOS_DIR','/kaggle/working/Ultralisk-v2.4'))
report24=json.loads((CACHE/'report.json').read_text())
assert report24['model']=='Ultralisk-v2.4 Chronos' and bool(report24['smoke'])==SMOKE
ANCHOR=report24['selected'] if report24['recommend_submit'] and not SMOKE else 'Baseline'
assert ANCHOR in ['Baseline']+METHODS
KEYS27=KEYS+['Regional','Narrow']; PREV=['RegionalFixed','RegionalAdaptive','RegionalReplace']; NEW=['NarrowFixed','NarrowAdaptive','NarrowReplace','NarrowStack','NarrowSpatial','NarrowCorrected']
MIN_GAIN=.002; STRESS_TOL=.002
log(f'Cache={CACHE}; anchor v2.4={ANCHOR}; smoke={SMOKE}')
for f in ['submission_ultralisk_v2_7.csv','submission_v2_7_NO_SUBIR.csv','submission_SMOKE_NO_SUBIR.csv']:
    (OUT/f).unlink(missing_ok=True)

def predict27(p,w):
    assert w.shape==(len(KEYS27),4) and np.isfinite(w).all() and (w>=-1e-7).all() and np.allclose(w.sum(0),1,atol=1e-5)
    return clip(sum(p[k]*w[i].astype('f4') for i,k in enumerate(KEYS27)))

def padded(w): return np.vstack([w,np.zeros((2,4),dtype='f4')])
def add_component(w,j,a):
    w=w*(1-a); w[j]+=a; return w

def calibration(p,y,method,h):
    w=padded(fit_method(p,y,'Baseline' if method=='V23' else ANCHOR))
    if method in ('V23','Anchor'): return w,None
    j=6 if method.startswith('Regional') else 7
    if method.endswith('Replace'):
        a=.75*(w[4]+w[5]); w[4:6]*=.25; w[j]+=a
    elif method.endswith('Fixed'): w=add_component(w,j,np.full(4,.20,dtype='f4'))
    elif method in ('NarrowAdaptive','RegionalAdaptive','NarrowSpatial','NarrowCorrected','NarrowStack'):
        a=fit_alpha(predict27(p,w),p[KEYS27[j]],y,prior=.20,shrink=.50,cap=.50)
        w=add_component(w,j,a)
    else: raise ValueError(method)
    if method=='NarrowStack':
        result=[]
        for v in range(4):
            prior=w[:,v].astype('f8'); prior=prior*.95; prior[6]+=.05
            # Regularizar frente a la mezcla jerarquica original, en unidades oficiales.
            e=np.stack([((p[k][...,v]-y[...,v])/NORM[v]).ravel() for k in KEYS27]).astype('f8'); gram=e@e.T/e.shape[1]; reg=.012
            bounds=[(max(0,float(x)-.18),min(1,float(x)+.18)) for x in prior]; bounds[7]=(0,.55)
            opt=minimize(lambda z:float(z@gram@z+reg*np.square(z-prior).sum()),prior,jac=lambda z:2*(gram@z+reg*(z-prior)),bounds=bounds,constraints=[{'type':'eq','fun':lambda z:z.sum()-1,'jac':lambda z:np.ones(len(KEYS27))}],method='SLSQP',options={'maxiter':100,'ftol':1e-11})
            q=opt.x if opt.success and np.isfinite(opt.x).all() and abs(opt.x.sum()-1)<1e-6 else prior; q=np.maximum(q,0); result.append(q/q.sum())
        w=np.array(result,dtype='f4').T
    maps=None
    if method in ('NarrowSpatial','NarrowCorrected'):
        # Dos alternativas: corregir el ensemble o solamente el kernel regional.
        e=y-(p['Narrow'] if method=='NarrowCorrected' else predict27(p,w)); glob=gaussian_filter(e.mean(0),sigma=(3,3,0)); maps=[]
        for hour in range(24):
            hw=np.exp(1.5*(np.cos((h-hour)*np.pi/12)-1)); local=np.average(e,axis=0,weights=hw)
            maps.append(.35*(.5*glob+.5*gaussian_filter(local,sigma=(3,3,0))))
        maps=np.asarray(maps,dtype='f4')
        if method=='NarrowCorrected': maps*=w[7]
    return w,maps

def apply27(p,params,h):
    w,b=params; pred=predict27(p,w); return clip(pred if b is None else pred+b[h])

ROWS=[]; CAL=[]; SCORES={}; REPRO=[]
def run_fold(label,cut,methods):
    cal,start,end=cut
    tr=np.flatnonzero(TT<cal); ca=np.flatnonzero((TT>=cal)&(TT<start)&np.isin(H,HOURS)); ev=np.flatnonzero((TT>=start)&(TT<end)&np.isin(H,HOURS))
    if SMOKE: tr,ca,ev=tr[:48],ca[:4],ev[:4]
    ix=np.r_[ca,ev]; n=len(ca); assert len(tr) and n and len(ev) and max(TT[tr])<min(TT[ca])<min(TT[ev])
    with np.load(CACHE/f'fold{label}_predictions.npz',allow_pickle=False) as f:
        assert np.array_equal(f['times'],TT[ix]) and int(f['n_cal'])==n and np.array_equal(f['target'],Y[ix]), 'Cache no corresponde al dataset o corte'
        p={k:f[k].astype('f4') for k in KEYS}
    assert all(v.shape==(len(ix),70,35,4) and np.isfinite(v).all() for v in p.values())
    st=stats(tr); t=time.perf_counter()
    p['Regional']=regional(DEVICES[0],st,tr,W[ix],H[ix],alpha=1.2,sigma=.65,local=.7,days=90)
    p['Narrow']=regional(DEVICES[0],st,tr,W[ix],H[ix],alpha=1.2,sigma=.4,local=.85,days=90)
    log(f'FOLD {label} regional kernels done {time.perf_counter()-t:.2f}s train={len(tr)} cal={n} eval={len(ev)}')
    pc={k:v[:n].copy() for k,v in p.items()}; pe={k:v[n:] for k,v in p.items()}; scores={}
    for m in dict.fromkeys(['V23','Anchor',*methods,'RegionalIndividual','NarrowIndividual']):
        params=None
        if m.endswith('Individual'): pred=pe[m[:-10]]
        else: params=calibration(pc,Y[ca],m,H[ca]); pred=apply27(pe,params,H[ev])
        s,r=score(pred,Y[ev]); scores[m]=s
        ROWS.append(dict(fold=str(label),model=m,skill=s,**dict(zip(['RMSE_'+v for v in VARS],r.tolist()))))
        log(f'FOLD {label} {m:20s} skill={s:.6f} RMSE={r.round(4).tolist()}')
    for m,old in [('V23','Baseline'),('Anchor',ANCHOR)]:
        expected=[r['skill'] for r in report24['validation'] if str(r['fold'])==str(label) and r['model']==old]
        assert len(expected)==1 and abs(scores[m]-expected[0])<2e-6, f'Cache no reproduce {m} fold {label}'
    if label in (1,2,3): REPRO.append(abs(scores['V23']-OLD_FOLD[label])<=REPRO_TOL); CAL.append((pc,Y[ca],H[ca]))
    SCORES[str(label)]=scores; pd.DataFrame(ROWS).to_csv(OUT/'validation_metrics.csv',index=False)

for f,cut in enumerate(CUTS[:2],1): run_fold(f,cut,PREV+NEW)
def choose(methods,base,min_gain):
    gains={m:[SCORES[str(f)][m]-max(SCORES[str(f)][b] for b in base) for f in (1,2)] for m in methods}
    eligible=[m for m,g in gains.items() if min(g)>=0 and np.mean(g)>=min_gain]
    return (max(eligible,key=lambda m:np.mean(gains[m])) if eligible else None),gains
prev,prev_gains=choose(PREV,['V23','Anchor'],.0015); prev=prev or 'Anchor'
baselines=list(dict.fromkeys(['V23','Anchor',prev]))
selected,dev_gains=choose(NEW,baselines,MIN_GAIN); selected=selected or prev
log(f'FROZEN before retrospective checks: previous={prev} selected={selected} gains={dev_gains}')
for f,cut in [(3,CUTS[2]),('April',STRESS)]: run_fold(f,cut,[prev,selected])
repro=all(REPRO) and bool(report24['reference_reproduced'])
audit_gain=SCORES['3'][selected]-max(SCORES['3'][b] for b in baselines); stress_gain=SCORES['April'][selected]-max(SCORES['April'][b] for b in baselines)
PASS=bool(not SMOKE and repro and selected in NEW and audit_gain>=MIN_GAIN and stress_gain>=-STRESS_TOL)
log(f'GATE repro={repro} audit_gain={audit_gain:+.6f} stress_gain={stress_gain:+.6f} recommend={PASS}')
# Los pesos finales usan exclusivamente las tres semanas de calibracion; no sus evaluaciones.
cp={k:np.concatenate([p[k] for p,y,h in CAL]) for k in KEYS27}; cy=np.concatenate([y for p,y,h in CAL]); ch=np.concatenate([h for p,y,h in CAL]); w,maps=calibration(cp,cy,selected,ch)
log(f'FINAL weights={dict(zip(KEYS27,w.round(5).tolist()))} spatial_bias={maps is not None}')
tr=np.argsort(TT); ev=np.unique(ti)
if SMOKE: tr=tr[:48]
with np.load(CACHE/'final_family_predictions.npz',allow_pickle=False) as f:
    assert np.array_equal(f['times'],TE[ev]); p={k:f[k].astype('f4') for k in KEYS}
assert all(v.shape==(len(ev),70,35,4) and np.isfinite(v).all() for v in p.values())
st=stats(tr)
p['Regional']=regional(DEVICES[0],st,tr,WT[ev],HT[ev],alpha=1.2,sigma=.65,local=.7,days=90,save_model=PASS,model_prefix='regional')
p['Narrow']=regional(DEVICES[0],st,tr,WT[ev],HT[ev],alpha=1.2,sigma=.4,local=.85,days=90,save_model=PASS,model_prefix='narrow')
forecast=apply27(p,(w,maps),HT[ev]); positions=np.full(len(TE),-1,dtype=int); positions[ev]=np.arange(len(ev)); assert (positions[ti]>=0).all()
sub=pd.DataFrame(forecast[positions[ti],ri,ci].round(4),columns=VARS); sub.insert(0,'id',TEMPLATE.id)
name='submission_SMOKE_NO_SUBIR.csv' if SMOKE else ('submission_ultralisk_v2_7.csv' if PASS else 'submission_v2_7_NO_SUBIR.csv'); dest=OUT/name; tmp=dest.with_suffix('.tmp'); sub.to_csv(tmp,index=False)
check=pd.read_csv(tmp,dtype={'id':str}); assert check.shape==TEMPLATE.shape and check.columns.tolist()==TEMPLATE.columns.tolist() and check.id.equals(TEMPLATE.id) and check.id.is_unique
assert np.isfinite(check[VARS].to_numpy()).all() and check.RH.between(0,100).all(); tmp.replace(dest)
np.savez(OUT/'calibration.npz',keys=np.asarray(KEYS27),weights=w,bias=np.zeros((0,),dtype='f4') if maps is None else maps)
report=dict(model='Ultralisk-v2.7',smoke=SMOKE,anchor=ANCHOR,previous=prev,selected=selected,reference_public_score=.49056,reference_reproduced=repro,dev_gains=dev_gains,previous_dev_gains=prev_gains,audit_gain=audit_gain,stress_gain=stress_gain,recommend_submit=PASS,keys=KEYS27,weights=w.tolist(),spatial_bias=maps is not None,validation=ROWS,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),rows=len(check),seconds=time.perf_counter()-START)
(OUT/'report.json').write_text(json.dumps(report,indent=2)); print(pd.DataFrame(ROWS).pivot(index='model',columns='fold',values='skill').round(6).to_string())
log(f'CSV={dest} rows={len(check)} SHA256={report["sha256"]}'); log('RECOMMEND_SUBMIT=YES' if PASS else 'RECOMMEND_SUBMIT=NO: conservar v2.3')
for f in (dest,OUT/'report.json',OUT/'validation_metrics.csv',LOG): display(FileLink(os.path.relpath(f)))
