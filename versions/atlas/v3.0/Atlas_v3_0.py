# Atlas v3.0 | single Kaggle cell | ShadowBytes
import os
from pathlib import Path
_cache=Path(os.environ.get('ULTRALISK_CHRONOS_DIR','/kaggle/working/Ultralisk-v2.4'))
_needed=['report.json','final_family_predictions.npz']+[f'fold{i}_predictions.npz' for i in (1,2,3,'April')]
if not all((_cache/f).is_file() for f in _needed):
    _cache=Path('/kaggle/working/Atlas-v3.0-cache')
    os.environ['ULTRALISK_CHRONOS_DIR']=str(_cache)
    if not all((_cache/f).is_file() for f in _needed):
        print('CACHE MISSING: rebuilding v2.4 neural families on T4 x2',flush=True)
        os.environ['ULTRALISK_OUT']=str(_cache)
        os.environ['ULTRALISK_CACHE']='1'
        exec(compile('# Ultralisk-v2.4 | ShadowBytes | ApuCast | Kaggle T4 x2\nimport os, time, json, copy, hashlib, threading, platform\nfrom pathlib import Path\nfrom concurrent.futures import ThreadPoolExecutor\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom IPython.display import display, FileLink\nSMOKE=os.environ.get(\'ULTRALISK_SMOKE\')==\'1\'  # prueba tecnica, NO subir\nOUT=Path(os.environ.get(\'ULTRALISK_OUT\',\'/kaggle/working/Ultralisk-v2.4\')); OUT.mkdir(parents=True,exist_ok=True)\nLOG=OUT/\'training.log\'; LOG.write_text(\'\'); LOCK=threading.Lock(); START=time.perf_counter()\nBATCH=32; EPOCHS_REF=25; EPOCHS_NEW=36\nSEEDS={\'CNN\':(0,1,2),\'ResNet\':(0,1,2),\'Mutalisk\':(42,123),\'Viper\':(42,123)}\nCUTS=[(\'2024050900\',\'2024051600\',\'2024060100\'),(\'2024052500\',\'2024060100\',\'2024061600\'),(\'2024060900\',\'2024061600\',\'2024070100\')]\nVARS=[\'RH\',\'T\',\'U\',\'V\']; NORM=np.array([25.360,2.0652,2.6776,2.0876])\n# Valores historicos de versions/v3/backtest.json (solo referencia, no etiquetas).\nHISTORICAL={1:.42292221906027727,2:.4136267261457105,3:.42723285641743414}\nMIN_DEV_GAIN=.003; MIN_AUDIT_GAIN=.002\ndef log(msg):\n    line=f\'[{time.perf_counter()-START:8.1f}s] {msg}\'\n    with LOCK:\n        print(line,flush=True)\n        with LOG.open(\'a\') as f: f.write(line+\'\\n\')\nif not SMOKE and (torch.cuda.device_count()!=2 or any(\'T4\' not in torch.cuda.get_device_name(i) for i in range(2))): raise RuntimeError(\'Selecciona Accelerator > GPU T4 x2\')\nDEVICES=[torch.device(\'cpu\')] if SMOKE else [torch.device(\'cuda:0\'),torch.device(\'cuda:1\')]\ntorch.set_num_threads(2); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True\ncpu_name=platform.processor()\nif Path(\'/proc/cpuinfo\').is_file():\n    cpu_name=next((l.split(\':\',1)[1].strip() for l in Path(\'/proc/cpuinfo\').read_text().splitlines() if l.startswith(\'model name\')),cpu_name)\nlog(f\'torch={torch.__version__} CPU={cpu_name} cores={os.cpu_count()} torch_threads={torch.get_num_threads()} smoke={SMOKE}\')\nfor d in DEVICES:\n    if d.type==\'cuda\': log(f\'{d} {torch.cuda.get_device_name(d)} VRAM={torch.cuda.get_device_properties(d).total_memory/2**30:.2f} GiB\')\n# Datos oficiales\nroot=Path(os.environ.get(\'ULTRALISK_DATA\',\'/kaggle/input\')); required=(\'train.npz\',\'test.npz\',\'static.npz\',\'sample_submission.csv\')\nfolders=[p.parent for p in root.rglob(\'train.npz\') if all((p.parent/k).is_file() for k in required)]\nofficial=[p for p in folders if \'hackathon-ia-pucp-2026\' in str(p)]\nif len(official)==1: DATA=official[0]\nelif len(folders)==1: DATA=folders[0]\nelse: raise RuntimeError(f\'Dataset oficial no identificado: {folders}\')\nwith np.load(DATA/\'train.npz\') as f: X,Y,TT=f[\'X\'].astype(\'f4\'),f[\'Y\'].astype(\'f4\'),f[\'times\'].astype(str)\nwith np.load(DATA/\'test.npz\') as f: XT,TE=f[\'X\'].astype(\'f4\'),f[\'times\'].astype(str)\nwith np.load(DATA/\'static.npz\') as f:\n    DEM=f[\'dem\'].astype(\'f4\'); glat,glon,wlat,wlon=[f[k] for k in (\'gfs_lats\',\'gfs_lons\',\'wrf_lats\',\'wrf_lons\')]\nTEMPLATE=pd.read_csv(DATA/\'sample_submission.csv\',dtype={\'id\':str})\nassert X.shape==(2935,17,13,4) and Y.shape==(2935,70,35,4) and XT.shape==(792,17,13,4) and DEM.shape==(70,35)\nassert TEMPLATE.shape==(646800,5) and TEMPLATE.columns.tolist()==[\'id\',*VARS] and TEMPLATE.id.is_unique\nassert all(np.isfinite(a).all() for a in (X,Y,XT,DEM)) and max(TT)<min(TE)\nparts=TEMPLATE.id.str.split(\'_\',expand=True); ids_t=parts[0].map({s:i for i,s in enumerate(TE)}); assert ids_t.notna().all()\nti=ids_t.to_numpy(dtype=int); ri=parts[1].astype(int).to_numpy(); ci=parts[2].astype(int).to_numpy()\nassert (ri>=0).all() and (ri<70).all() and (ci>=0).all() and (ci<35).all()\nassert len(np.unique(ti))==264 and np.all(np.bincount(ti)[np.unique(ti)]==2450)\nHOURS=np.unique([int(s[-2:]) for s in parts[0].unique()]); H=np.array([int(s[-2:]) for s in TT]); HT=np.array([int(s[-2:]) for s in TE])\nlog(f\'Data={DATA} train={X.shape} test={XT.shape} horas={HOURS.tolist()}\')\ndef windows(x,times):\n    dates=pd.to_datetime(times,format=\'%Y%m%d%H\'); lookup={t:i for i,t in enumerate(dates)}\n    return x[np.array([[lookup.get(t-pd.Timedelta(hours=k),i) for k in (2,1,0)] for i,t in enumerate(dates)])]\nW,WT=windows(X,TT),windows(XT,TE)\nlat,lon=np.meshgrid(wlat,wlon,indexing=\'ij\'); grid=np.stack([(lon-glon[0])/(glon[-1]-glon[0])*2-1,(lat-glat[0])/(glat[-1]-glat[0])*2-1],-1)[None].astype(\'f4\')\ndef z(a): return ((a-a.mean())/max(float(a.std()),1e-6)).astype(\'f4\')\ndef smooth(a,k):\n    p=k//2; return F.avg_pool2d(F.pad(torch.from_numpy(a[None,None]),(p,p,p,p),mode=\'replicate\'),k,stride=1)[0,0].numpy()\ndy,dx=np.gradient(DEM)\ngeo=np.stack([z(DEM),z(dx),z(dy),z(np.hypot(dx,dy)),z(DEM-smooth(DEM,3)),z(DEM-smooth(DEM,7)),z(DEM-smooth(DEM,15)),z(lat),z(lon),z(np.gradient(dy,axis=0)+np.gradient(dx,axis=1))])[None].astype(\'f4\')\ndef stats(ix):\n    yy,xx,hh=Y[ix],X[ix],H[ix]\n    def climate(a): return np.stack([a[hh==h].mean(0) if np.any(hh==h) else a.mean(0) for h in range(24)])\n    cl=climate(yy)\n    return dict(mu=xx.mean((0,1,2)),sd=np.maximum(xx.std((0,1,2)),1e-6),clim=cl,gfs_clim=climate(xx),scale=np.maximum((yy-cl[hh]).std((0,1,2)),1e-6))\ndef score(p,y):\n    rm=np.sqrt(np.mean(np.square(p.astype(\'f8\')-y),axis=(0,1,2))); return float(1-(rm/NORM).mean()),rm\ndef clip(p):\n    p=p.astype(\'f4\',copy=False); p[...,0]=p[...,0].clip(0,100); assert np.isfinite(p).all(); return p\ndef tensor(a,d): return torch.as_tensor(a,dtype=torch.float32,device=d)\ndef sync(d):\n    if d.type==\'cuda\': torch.cuda.synchronize(d)\n# Arquitecturas\nclass Residual(nn.Module):\n    def __init__(self,a,b,dilation=1):\n        super().__init__(); self.c1=nn.Conv2d(a,b,3,padding=dilation,dilation=dilation); self.c2=nn.Conv2d(b,b,3,padding=1); self.skip=nn.Conv2d(a,b,1) if a!=b else nn.Identity()\n    def forward(self,x): return F.gelu(self.c2(F.gelu(self.c1(x)))+self.skip(x))\ndef inputs(x,h):\n    n=len(x); a=h*(2*np.pi/24); clock=torch.stack([a.sin(),a.cos()],1)[:,:,None,None].expand(-1,-1,17,13)\n    return x.permute(0,1,4,2,3).reshape(n,12,17,13),clock\nclass Reference(nn.Module):\n    def __init__(self,kind):\n        super().__init__(); w=64\n        self.enc=nn.Sequential(nn.Conv2d(14,w,3,padding=1),nn.GELU(),nn.Conv2d(w,w,3,padding=1),nn.GELU())\n        self.static=nn.Parameter(torch.zeros(1,8,70,35))\n        self.dec=nn.Sequential(nn.Conv2d(73,w,3,padding=1),nn.GELU(),nn.Conv2d(w,w,3,padding=1),nn.GELU(),nn.Conv2d(w,4,1))\n        if kind==\'ResNet\': self.enc=Residual(14,w); self.dec=nn.Sequential(Residual(73,w),nn.Conv2d(w,4,1))\n        self.register_buffer(\'grid\',torch.from_numpy(grid)); self.register_buffer(\'dem\',torch.from_numpy(z(DEM)[None,None]))\n    def forward(self,x,h):\n        n=len(x); fields,clock=inputs(x,h); v=self.enc(torch.cat([fields,clock],1))\n        with torch.autocast(device_type=x.device.type,enabled=False): v=F.grid_sample(v.float(),self.grid.expand(n,-1,-1,-1),align_corners=True)\n        return self.dec(torch.cat([v,self.dem.expand(n,-1,-1,-1),self.static.expand(n,-1,-1,-1)],1)).permute(0,2,3,1)\nclass GeoWind(nn.Module):\n    def __init__(self,kind,st):\n        super().__init__(); self.kind=kind; w=48\n        self.register_buffer(\'grid\',torch.from_numpy(grid)); self.register_buffer(\'geo\',torch.from_numpy(geo)); self.register_buffer(\'gcl\',torch.from_numpy(((st[\'gfs_clim\']-st[\'mu\'])/st[\'sd\']).astype(\'f4\')))\n        self.enc=nn.Sequential(Residual(22,w),Residual(w,w,2),Residual(w,w)); self.map=nn.Parameter(torch.zeros(1,8,70,35)); self.fuse=Residual(w+26,w)\n        if kind==\'Mutalisk\': self.down=nn.Sequential(Residual(w,64),Residual(64,64,2)); self.up=Residual(w+64,w)\n        else: self.wind=nn.Sequential(Residual(w,w,2),Residual(w,w,3))\n        self.thermal=nn.Sequential(Residual(w,32),nn.Conv2d(32,2,1)); self.velocity=nn.Sequential(Residual(w,32),nn.Conv2d(32,2,1))\n    def interp(self,x):\n        with torch.autocast(device_type=x.device.type,enabled=False): return F.grid_sample(x.float(),self.grid.expand(len(x),-1,-1,-1),align_corners=True,padding_mode=\'border\')\n    def forward(self,x,h):\n        n=len(x); fields,clock=inputs(x,h); current=x[:,-1]; anomaly=current-self.gcl[h.long()]; trend=current-x[:,0]\n        av=anomaly.permute(0,3,1,2); cv=current.permute(0,3,1,2); v=self.enc(torch.cat([fields,av,trend.permute(0,3,1,2),clock],1))\n        fine=self.fuse(torch.cat([self.interp(v),self.interp(torch.cat([cv,av],1)),self.geo.expand(n,-1,-1,-1),self.map.expand(n,-1,-1,-1)],1))\n        if self.kind==\'Mutalisk\':\n            low=self.down(F.avg_pool2d(fine,2)); fine=self.up(torch.cat([fine,F.interpolate(low,size=fine.shape[-2:],mode=\'bilinear\',align_corners=False)],1)); wind=fine\n        else: wind=self.wind(fine)\n        return torch.cat([self.thermal(fine),self.velocity(wind)],1).permute(0,2,3,1)\ndef make(kind,st): return Reference(kind) if kind in (\'CNN\',\'ResNet\') else GeoWind(kind,st)\n# Benchmark de inferencia, mismo modelo y lote para todos los dispositivos.\ndef benchmark():\n    st=stats(np.arange(128)); torch.manual_seed(0); base=Reference(\'ResNet\').eval(); results=[]\n    x=(W[:8]-st[\'mu\'])/st[\'sd\']; h=H[:8]\n    for d,amp in [(torch.device(\'cpu\'),False)]+[(d,a) for d in DEVICES if d.type==\'cuda\' for a in (False,True)]:\n        m=copy.deepcopy(base).to(d,memory_format=torch.channels_last); xb,hb=tensor(x,d),tensor(h,d)\n        with torch.inference_mode():\n            for _ in range(3):\n                with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=amp): m(xb,hb)\n            sync(d); t=time.perf_counter()\n            for _ in range(8):\n                with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=amp): m(xb,hb)\n            sync(d); sec=time.perf_counter()-t\n        r=dict(device=str(d),amp=amp,batch=8,ms_per_batch=sec/8*1000,samples_per_second=64/sec); results.append(r); log(f\'BENCHMARK {r}\')\n    if len(DEVICES)==2:\n        barrier=threading.Barrier(2)\n        def both(d):\n            m=copy.deepcopy(base).to(d,memory_format=torch.channels_last); xb,hb=tensor(x,d),tensor(h,d)\n            with torch.inference_mode(), torch.autocast(device_type=\'cuda\',dtype=torch.float16):\n                for _ in range(3): m(xb,hb)\n                sync(d); barrier.wait(timeout=60); t=time.perf_counter()\n                for _ in range(8): m(xb,hb)\n                sync(d); return time.perf_counter()-t\n        with ThreadPoolExecutor(max_workers=2) as pool:\n            sec=max(pool.map(both,DEVICES))\n        r=dict(device=\'T4x2_parallel\',amp=True,batch=16,ms_per_batch=sec/8*1000,samples_per_second=128/sec)\n        results.append(r); log(f\'BENCHMARK {r}\')\n    return results\nBENCH=benchmark() if os.environ.get(\'ULTRALISK_BENCH\',\'0\')==\'1\' else []\nlog(\'Benchmark opcional; iniciando experimentos v2.4\')\n@torch.inference_mode()\ndef infer(model,x,h,d,amp):\n    model.eval(); out=[]\n    for i in range(0,len(x),BATCH):\n        with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=amp): p=model(x[i:i+BATCH],h[i:i+BATCH])\n        out.append(p.float().cpu().numpy())\n    return np.concatenate(out)\ndef fit_neural(job,d,st,tr,we,he,tag):\n    kind,seed,model=job; ref=kind in (\'CNN\',\'ResNet\'); epochs=1 if SMOKE else (EPOCHS_REF if ref else EPOCHS_NEW)\n    amp=not ref and d.type==\'cuda\'; lr=.002 if ref else .0012; rng=np.random.RandomState(seed)\n    model=model.to(d,memory_format=torch.channels_last); ema=None if ref else copy.deepcopy(model).eval().requires_grad_(False)\n    x=tensor((W[tr]-st[\'mu\'])/st[\'sd\'],d); h=tensor(H[tr],d); y=tensor((Y[tr]-st[\'clim\'][H[tr]])/st[\'scale\'],d)\n    xe=tensor((we-st[\'mu\'])/st[\'sd\'],d); he_gpu=tensor(he,d)\n    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4 if ref else 5e-4)\n    steps=(len(tr)+BATCH-1)//BATCH; sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps*epochs)\n    scaler=torch.amp.GradScaler(\'cuda\',enabled=amp); ratio=tensor(st[\'scale\']/NORM,d)\n    order=np.arange(len(tr)); started=time.perf_counter(); hist=[]\n    if d.type==\'cuda\': torch.cuda.reset_peak_memory_stats(d)\n    log(f\'{tag} {kind}:{seed} {d} START params={sum(p.numel() for p in model.parameters()):,} epochs={epochs} amp={amp}\')\n    for ep in range(1,epochs+1):\n        model.train(); rng.shuffle(order); accum=torch.zeros((),device=d); t0=time.perf_counter()\n        for k in range(0,len(order),BATCH):\n            ix=torch.as_tensor(order[k:k+BATCH],device=d,dtype=torch.long); opt.zero_grad(set_to_none=True)\n            with torch.autocast(device_type=d.type,dtype=torch.float16,enabled=amp): p=model(x[ix],h[ix])\n            mse=(p.float()-y[ix]).square(); loss=mse.mean() if ref else .5*mse.mean()+.5*((mse.mean((0,1,2))+1e-8).sqrt()*ratio).mean()\n            scaler.scale(loss).backward()\n            if not ref: scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.)\n            old=scaler.get_scale(); scaler.step(opt); scaler.update()\n            if scaler.get_scale()>=old:\n                sched.step()\n                if ema is not None:\n                    with torch.no_grad():\n                        for a,b in zip(ema.parameters(),model.parameters()): a.lerp_(b,.015)\n            accum+=loss.detach()*len(ix)\n        sync(d); sec=time.perf_counter()-t0; value=float(accum.item()/len(tr)); assert np.isfinite(value)\n        mem=torch.cuda.max_memory_allocated(d)/2**30 if d.type==\'cuda\' else None\n        hist.append(dict(epoch=ep,loss=value,seconds=sec,samples_per_second=len(tr)/sec,peak_gib=mem,lr=opt.param_groups[0][\'lr\']))\n        log(f\'{tag} {kind}:{seed} {d} ep={ep:02d}/{epochs} loss={value:.5f} lr={opt.param_groups[0]["lr"]:.6f} {sec:.2f}s {len(tr)/sec:.1f} samples/s VRAM={mem}\')\n    chosen=model if ref else ema; p=clip(infer(chosen,xe,he_gpu,d,amp)*st[\'scale\']+st[\'clim\'][he]); elapsed=time.perf_counter()-started\n    pd.DataFrame(hist).to_csv(OUT/f\'{tag}_{kind}_{seed}_history.csv\',index=False)\n    if tag==\'final\': torch.save({\'kind\':kind,\'seed\':seed,\'epochs\':epochs,\'state_dict\':{k:v.detach().cpu() for k,v in chosen.state_dict().items()}},OUT/f\'{kind}_{seed}.pt\')\n    log(f\'{tag} {kind}:{seed} DONE {elapsed:.1f}s\'); model.to(\'cpu\')\n    return kind,p,dict(kind=kind,seed=seed,device=str(d),seconds=elapsed,epochs=epochs,peak_gib=mem)\n# Kernel Ridge no lineal: anomalias del GFS, tendencia y promedio temporal.\ndef kernel_features(w,h,st):\n    a=(w-st[\'gfs_clim\'][(h[:,None]-np.array([2,1,0]))%24])/st[\'sd\']; a=a.reshape(len(w),3,-1)\n    return np.concatenate([a[:,2],a.mean(1),.5*(a[:,2]-a[:,0])],axis=1).astype(\'f4\')\ndef kernel(a,b):\n    dot=(a@b.T)/a.shape[1]; dist=(a.square().mean(1)[:,None]+b.square().mean(1)[None,:]-2*dot).clamp_min(0)\n    return .5*dot+.25*dot.square()+.25*torch.exp(-dist/2)\ndef fit_kernel(d,st,tr,we,he,tag):\n    t=time.perf_counter(); a=kernel_features(W[tr],H[tr],st); b=kernel_features(we,he,st)\n    mu=a.mean(0); sd=np.maximum(a.std(0),.15); a=(a-mu)/sd; b=(b-mu)/sd\n    ag,bg=tensor(a,d),tensor(b,d); K=kernel(ag,ag); K=(K+K.T)*.5; K.diagonal().add_(.05)\n    L=torch.linalg.cholesky(K); yy=tensor(((Y[tr]-st[\'clim\'][H[tr]])/st[\'scale\']).reshape(len(tr),-1),d)\n    dual=torch.cholesky_solve(yy,L); p=(kernel(bg,ag)@dual).cpu().numpy().reshape(len(we),70,35,4)\n    if tag==\'final\': np.savez_compressed(OUT/\'kernel_model.npz\',features=a,mu=mu,sd=sd,dual=dual.cpu().numpy())\n    log(f\'{tag} Kernel DONE {time.perf_counter()-t:.2f}s alpha=.05\'); return clip(p*st[\'scale\']+st[\'clim\'][he])\nALL_RUNS=[]\ndef run_stage(tr,we,he,tag,families):\n    st=stats(tr); jobs=[]\n    for kind in families:\n        if kind==\'Kernel\': continue\n        for seed in (SEEDS[kind][:1] if SMOKE else SEEDS[kind]):\n            torch.manual_seed(seed); jobs.append((kind,seed,make(kind,st)))\n    def worker(group,d): return [fit_neural(j,d,st,tr,we,he,tag) for j in group]\n    results=[]\n    if len(DEVICES)==2:\n        with ThreadPoolExecutor(max_workers=2) as pool:\n            futures=[pool.submit(worker,jobs[i::2],d) for i,d in enumerate(DEVICES)]\n            for f in futures: results.extend(f.result())\n    else: results=worker(jobs,DEVICES[0])\n    ALL_RUNS.extend([dict(tag=tag,**r) for _,_,r in results]); results.sort(key=lambda x:x[2][\'seed\'])\n    pred={k:np.mean([p for kind,p,_ in results if kind==k],axis=0,dtype=np.float32) for k in families if k!=\'Kernel\'}\n    if \'Kernel\' in families: pred[\'Kernel\']=fit_kernel(DEVICES[0],st,tr,we,he,tag)\n    if \'Kernel\' in families: pred[\'Clock\']=fit_clock(DEVICES[0],st,tr,we,he,tag)\n    if tag==\'final\': np.savez_compressed(OUT/\'preprocessing.npz\',**st,grid=grid,geo=geo,normalizers=NORM)\n    return pred\n# Calibracion por variable, separada de evaluacion.\nCANDIDATES={\'Reference\':{},\'Mutalisk\':{\'Mutalisk\':1.},\'Viper\':{\'Viper\':1.},\'Neural\':{\'Mutalisk\':.5,\'Viper\':.5},\'Kernel\':{\'Kernel\':1.},\'Hybrid\':{\'Mutalisk\':.375,\'Viper\':.375,\'Kernel\':.25}}\ndef fit_alpha(a,b,y,prior=.5,shrink=.2,cap=1.):\n    d=b.astype(\'f8\')-a; r=y.astype(\'f8\')-a; den=(d*d).mean((0,1,2)); num=(r*d).mean((0,1,2))\n    raw=np.divide(num,den,out=np.full(4,prior),where=den>1e-12); return np.clip((1-shrink)*raw+shrink*prior,0,cap).astype(\'f4\')\ndef mix(a,b,w): return clip(a*(1-w)+b*w)\ndef candidate(p,name): return sum(p[k]*np.float32(w) for k,w in CANDIDATES[name].items())\ndef calibrate(p,y,name):\n    rw=fit_alpha(p[\'CNN\'],p[\'ResNet\'],y); ref=mix(p[\'CNN\'],p[\'ResNet\'],rw)\n    alpha=np.zeros(4,dtype=\'f4\') if name==\'Reference\' else fit_alpha(ref,candidate(p,name),y,prior=.25,shrink=.35,cap=.65)\n    return rw,alpha\ndef combine(p,name,rw,alpha):\n    ref=mix(p[\'CNN\'],p[\'ResNet\'],rw); return ref if name==\'Reference\' else mix(ref,candidate(p,name),alpha)\n\n\n# ================================================================\n# ULTRALISK v2.4 | CHRONOS | extension incremental de v2.3\n# No usa WRF oculto, datos externos ni horas futuras de GFS.\n# ================================================================\nfrom scipy.optimize import minimize\nPUBLIC_REFERENCE=.49056\nOLD_FOLD={1:.439850572420165,2:.4396332592736687,3:.4496978946404211}\nSTRESS=(\'2024040900\',\'2024041600\',\'2024050100\')\n# Dos variantes conservadoras, probadas individualmente en CPU en folds 1 y 2.\nCLOCK_SETTINGS=[(.6,90),(1.2,90)]\nFAMILIES=[\'CNN\',\'ResNet\',\'Mutalisk\',\'Viper\',\'Kernel\']\nKEYS=FAMILIES+[\'Clock\']\nMETHODS=[\'ClockReplace\',\'ClockBlend\',\'Stack\']\nMIN_DEV=.002; MIN_AUDIT=.002; STRESS_TOL=.002; REPRO_TOL=.0015\n\ndef fit_clock(d,st,tr,we,he,tag):\n    t=time.perf_counter()\n    a=kernel_features(W[tr],H[tr],st); b=kernel_features(we,he,st)\n    mu=a.mean(0); sd=np.maximum(a.std(0),.15)\n    a=(a-mu)/sd; b=(b-mu)/sd\n    ag,bg=tensor(a,d),tensor(b,d)\n    h,eh=tensor(H[tr],d),tensor(he,d)\n    def clock(a,b): return .4+.6*(1+torch.cos((a[:,None]-b[None,:])*(np.pi/12)))/2\n    K=kernel(ag,ag)*clock(h,h); Q=kernel(bg,ag)*clock(eh,h)\n    K=(K+K.T)*.5\n    stamps=pd.to_datetime(TT[tr],format=\'%Y%m%d%H\')\n    age=tensor(np.asarray((stamps.max()-stamps).total_seconds()/86400,dtype=\'f4\'),d)\n    # Edad calculada solo desde fechas del entrenamiento. Un dia mas aproxima\n    # el comienzo del siguiente periodo de calibracion (igual en todos los folds).\n    age=age+1.\n    target=tensor(((Y[tr]-st[\'clim\'][H[tr]])/st[\'scale\']).reshape(len(tr),-1),d)\n    preds=[]\n    for alpha,days in CLOCK_SETTINGS:\n        Kreg=K.clone(); Kreg.diagonal().add_(alpha*torch.exp(age/days))\n        L=torch.linalg.cholesky(Kreg); dual=torch.cholesky_solve(target,L)\n        p=(Q@dual).cpu().numpy().reshape(len(we),70,35,4)\n        preds.append(clip(p*st[\'scale\']+st[\'clim\'][he]))\n        if tag==\'final\':\n            np.savez_compressed(OUT/f\'clock_alpha_{alpha}.npz\',features=a,mu=mu,sd=sd,dual=dual.cpu().numpy(),hours=H[tr],alpha=alpha,days=days)\n        del Kreg,L,dual,p\n        log(f\'{tag} Clock alpha={alpha} days={days} done elapsed={time.perf_counter()-t:.2f}s\')\n    return clip(np.mean(preds,axis=0,dtype=np.float32))\n\ndef old_weights(p,y):\n    rw,alpha=calibrate(p,y,\'Hybrid\')\n    # Convertir mezcla jerarquica v2.3 en pesos efectivos por variable.\n    return np.stack([(1-alpha)*(1-rw),(1-alpha)*rw,.375*alpha,.375*alpha,.25*alpha,np.zeros(4,dtype=\'f4\')])\n\ndef weighted(p,w):\n    return clip(sum(p[k]*w[i].astype(\'f4\') for i,k in enumerate(KEYS)))\n\ndef fit_method(p,y,method):\n    w=old_weights(p,y)\n    if method==\'Baseline\': return w\n    if method==\'ClockReplace\':\n        # Misma arquitectura y calibracion que v2.3, sustituyendo 75% del\n        # componente kernel por la variante horaria/reciente.\n        ref=mix(p[\'CNN\'],p[\'ResNet\'],fit_alpha(p[\'CNN\'],p[\'ResNet\'],y))\n        fresh=.375*p[\'Mutalisk\']+.375*p[\'Viper\']+.0625*p[\'Kernel\']+.1875*p[\'Clock\']\n        alpha=fit_alpha(ref,fresh,y,prior=.25,shrink=.35,cap=.65)\n        rw=fit_alpha(p[\'CNN\'],p[\'ResNet\'],y)\n        return np.stack([(1-alpha)*(1-rw),(1-alpha)*rw,.375*alpha,.375*alpha,.0625*alpha,.1875*alpha])\n    if method==\'ClockBlend\':\n        a=fit_alpha(weighted(p,w),p[\'Clock\'],y,prior=.15,shrink=.4,cap=.55)\n        w=w*(1-a); w[-1]+=a; return w\n    if method!=\'Stack\': raise ValueError(method)\n    # Regresion convexa multisalida. Se ajusta solo en la semana de calibracion.\n    # Penalizacion en escala de RMSE oficial y limites para evitar pesos extremos.\n    out=[]\n    for v in range(4):\n        prior=w[:,v].astype(\'f8\')*.85; prior[-1]+=.15\n        # Residuos normalizados, matrices de solo 6x6.\n        a=np.stack([((p[k][...,v]-y[...,v])/NORM[v]).reshape(-1) for k in KEYS],axis=0).astype(\'f8\')\n        gram=(a@a.T)/a.shape[1]; reg=.004\n        def fun(z): return float(z@gram@z+reg*np.square(z-prior).sum())\n        def jac(z): return 2*(gram@z+reg*(z-prior))\n        bounds=[(max(0,float(t)-.25),min(1,float(t)+.25)) for t in prior]; bounds[-1]=(0,.55)\n        opt=minimize(fun,prior,jac=jac,bounds=bounds,constraints=[dict(type=\'eq\',fun=lambda z:z.sum()-1,jac=lambda z:np.ones(6))],method=\'SLSQP\',options=dict(maxiter=100,ftol=1e-11))\n        z=opt.x if opt.success and np.isfinite(opt.x).all() and abs(opt.x.sum()-1)<1e-6 else prior\n        z=np.maximum(z,0); out.append(z/z.sum())\n    return np.asarray(out,dtype=\'f4\').T\n\n# Cache de predicciones de calibracion y validacion para futuras versiones.\n# Las etiquetas se obtienen exclusivamente de train.npz.\ndef fold_run(label,cut,methods):\n    cal,start,end=cut\n    tr=np.flatnonzero(TT<cal); ca=np.flatnonzero((TT>=cal)&(TT<start)&np.isin(H,HOURS)); ev=np.flatnonzero((TT>=start)&(TT<end)&np.isin(H,HOURS))\n    if SMOKE: tr,ca,ev=tr[:48],ca[:4],ev[:4]\n    assert len(tr) and len(ca) and len(ev) and max(TT[tr])<min(TT[ca])<min(TT[ev])\n    log(f\'FOLD {label} train={len(tr)} cal={len(ca)} eval={len(ev)} {cal}/{start}/{end}\')\n    ix=np.r_[ca,ev]; p=run_stage(tr,W[ix],H[ix],f\'fold{label}\',FAMILIES)\n    pc={k:v[:len(ca)].copy() for k,v in p.items()}; pe={k:v[len(ca):] for k,v in p.items()}\n    scores={}; rows=[]\n    for method in methods:\n        weights=fit_method(pc,Y[ca],method); pred=weighted(pe,weights); s,r=score(pred,Y[ev]); scores[method]=s\n        rows.append(dict(fold=str(label),model=method,skill=s,**dict(zip([\'RMSE_\'+v for v in VARS],r.tolist()))))\n        log(f\'FOLD {label} {method:13s} skill={s:.6f} RMSE={np.round(r,4).tolist()} weights={weights.round(3).tolist()}\')\n    if os.environ.get(\'ULTRALISK_CACHE\',\'1\')==\'1\':\n        np.savez(OUT/f\'fold{label}_predictions.npz\',times=TT[ix],n_cal=len(ca),target=Y[ix],**p)\n    return scores,rows,(pc,Y[ca])\n\nROWS=[]; CAL=[]; DEV=[]\nfor f,cut in enumerate(CUTS[:2],1):\n    sc,rows,cal=fold_run(f,cut,[\'Baseline\']+METHODS); DEV.append(sc); ROWS+=rows; CAL.append(cal)\n    pd.DataFrame(ROWS).to_csv(OUT/\'validation_metrics.csv\',index=False)\ndev_gains={m:[s[m]-s[\'Baseline\'] for s in DEV] for m in METHODS}\nrepro=all(abs(s[\'Baseline\']-OLD_FOLD[i])<=REPRO_TOL for i,s in enumerate(DEV,1))\neligible=[m for m,g in dev_gains.items() if min(g)>=0 and np.mean(g)>=MIN_DEV]\nselected=max(eligible,key=lambda m:np.mean(dev_gains[m])) if eligible else \'Baseline\'\nlog(f\'FROZEN BEFORE CHECKS selected={selected} gains={dev_gains} baseline_reproduced={repro}\')\nmethods=list(dict.fromkeys([\'Baseline\',selected]))\n# Junio III ya fue observado durante el desarrollo de v2.3: regresion historica,\n# no un nuevo test ciego. Abril es un corte adicional de estres cronologico.\naudit,rows,cal=fold_run(3,CUTS[2],methods); ROWS+=rows; CAL.append(cal)\npd.DataFrame(ROWS).to_csv(OUT/\'validation_metrics.csv\',index=False)\nrepro=repro and abs(audit[\'Baseline\']-OLD_FOLD[3])<=REPRO_TOL\nstress,rows,_=fold_run(\'April\',STRESS,methods); ROWS+=rows\npd.DataFrame(ROWS).to_csv(OUT/\'validation_metrics.csv\',index=False)\ngain=audit[selected]-audit[\'Baseline\']; stress_gain=stress[selected]-stress[\'Baseline\']\nPASS=selected!=\'Baseline\' and repro and gain>=MIN_AUDIT and stress_gain>=-STRESS_TOL and not SMOKE\nlog(f\'GATE selected={selected} june_gain={gain:+.6f} april_gain={stress_gain:+.6f} reproduced={repro} recommend={PASS}\')\n# La calibracion final concatena solo las tres semanas originales, no evaluaciones.\n# No se usa el puntaje publico como etiqueta ni para optimizar pesos.\ncal_p={k:np.concatenate([p[k] for p,y in CAL]) for k in KEYS}; cal_y=np.concatenate([y for p,y in CAL]); weights=fit_method(cal_p,cal_y,selected)\nlog(f\'FINAL weights keys={KEYS} weights={weights.tolist()}\')\ntr=np.argsort(TT); eval_t=np.unique(ti)\nif SMOKE: tr=tr[:48]\np=run_stage(tr,WT[eval_t],HT[eval_t],\'final\',FAMILIES)\nforecast=weighted(p,weights)\npositions=np.full(len(TE),-1,dtype=int); positions[eval_t]=np.arange(len(eval_t))\nvalues=forecast[positions[ti],ri,ci]; sub=pd.DataFrame(values.round(4),columns=VARS); sub.insert(0,\'id\',TEMPLATE.id)\nname=\'submission_ultralisk_v2_4.csv\' if PASS else \'submission_v2_4_NO_SUBIR.csv\'\nif SMOKE: name=\'submission_SMOKE_NO_SUBIR.csv\'\ndest=OUT/name; tmp=dest.with_suffix(\'.tmp\'); sub.to_csv(tmp,index=False)\ncheck=pd.read_csv(tmp,dtype={\'id\':str}); assert check.shape==TEMPLATE.shape and check.columns.tolist()==TEMPLATE.columns.tolist() and check.id.equals(TEMPLATE.id) and check.id.is_unique\nassert np.isfinite(check[VARS].to_numpy()).all() and check.RH.between(0,100).all(); tmp.replace(dest)\nif os.environ.get(\'ULTRALISK_CACHE\',\'1\')==\'1\': np.savez(OUT/\'final_family_predictions.npz\',times=TE[eval_t],**p)\nreport=dict(model=\'Ultralisk-v2.4 Chronos\',smoke=SMOKE,selected=selected,reference_public_score=PUBLIC_REFERENCE,reference_reproduced=repro,dev_gains=dev_gains,audit_gain=gain,stress_gain=stress_gain,recommend_submit=bool(PASS),keys=KEYS,weights=weights.tolist(),clock_settings=CLOCK_SETTINGS,validation=ROWS,benchmark=BENCH,runs=ALL_RUNS,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),rows=len(check),seconds=time.perf_counter()-START)\n(OUT/\'report.json\').write_text(json.dumps(report,indent=2))\nprint(pd.DataFrame(ROWS).pivot(index=\'model\',columns=\'fold\',values=\'skill\').round(6).to_string())\nlog(f\'CSV={dest} SHA256={report["sha256"]} rows={len(check)}\')\nlog(\'RECOMMEND_SUBMIT=YES\' if PASS else \'RECOMMEND_SUBMIT=NO: conservar CSV v2.3 (0.49056)\')\nfor f in (dest,OUT/\'report.json\',OUT/\'validation_metrics.csv\',LOG): display(FileLink(os.path.relpath(f)))\n', 'embedded_ultralisk_v24.py','exec'),{'__name__':'__main__'})
else:
    print('Reusing v2.4 prediction cache',flush=True)
# Atlas-v3.0 | ShadowBytes | extended meteorological memory
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
OUT=Path(os.environ.get('ATLAS_OUT','/kaggle/working/Atlas-v3.0')); OUT.mkdir(parents=True,exist_ok=True)
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


# Atlas v3.0 -- compare against the actual v2.7 calibration, never submit automatically.
from scipy.ndimage import gaussian_filter
CACHE=Path(os.environ.get('ULTRALISK_CHRONOS_DIR','/kaggle/working/Ultralisk-v2.4'))
report24=json.loads((CACHE/'report.json').read_text())
assert report24['model']=='Ultralisk-v2.4 Chronos' and bool(report24['smoke'])==SMOKE
ANCHOR=report24['selected'] if report24['recommend_submit'] and not SMOKE else 'Baseline'
assert ANCHOR in ['Baseline']+METHODS
KEYS27=KEYS+['Regional','Narrow']

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

# v2.7's selected method was NarrowSpatial; use the exact same method in every fold.
BASE='NarrowSpatial'
NAMES=['History','WindHistory','WindOnly','EqualHistory']
EXPECTED={'1':.45990063168146056,'2':.4604174971675282,'3':.4650797440837502,'April':.5584575773496497}

def bias_maps(p,y,h):
    e=y-p; glob=gaussian_filter(e.mean(0),sigma=(3,3,0)); maps=[]
    for hour in range(24):
        hw=np.exp(1.5*(np.cos((h-hour)*np.pi/12)-1)); local=np.average(e,axis=0,weights=hw)
        maps.append(.35*(.5*glob+.5*gaussian_filter(local,sigma=(3,3,0))))
    return np.asarray(maps,dtype='f4')

def candidate_atlas(pc,y,h,method):
    base=calibration(pc,y,BASE,h); a=apply27(pc,base,h)
    if method=='Reference': return base,None,None,None
    key='WindHistory' if method in ('WindHistory','WindOnly') else 'History'
    if method=='EqualHistory': b=.5*(pc['History']+pc['WindHistory'])
    else: b=pc[key]
    bm=bias_maps(b,y,h); b=clip(b+bm[h]); w=fit_alpha(a,b,y,prior=.20,shrink=.5,cap=.55)
    if method=='WindOnly': w[:2]=0
    return base,bm,w,method

def apply_atlas(p,params,h):
    base,bm,w,method=params; a=apply27(p,base,h)
    if method is None: return a
    b=.5*(p['History']+p['WindHistory']) if method=='EqualHistory' else p['WindHistory' if method in ('WindHistory','WindOnly') else 'History']
    return mix(a,clip(b+bm[h]),w)

def load_preds(label,cut,methods):
    cal,start,end=cut; tr=np.flatnonzero(TT<cal); ca=np.flatnonzero((TT>=cal)&(TT<start)&np.isin(H,HOURS)); ev=np.flatnonzero((TT>=start)&(TT<end)&np.isin(H,HOURS))
    if SMOKE: tr,ca,ev=tr[:48],ca[:4],ev[:4]
    ix=np.r_[ca,ev]; n=len(ca); assert len(tr) and n and len(ev) and max(TT[tr])<min(TT[ca])<min(TT[ev])
    with np.load(CACHE/f'fold{label}_predictions.npz',allow_pickle=False) as f:
        assert np.array_equal(f['times'],TT[ix]) and int(f['n_cal'])==n and np.array_equal(f['target'],Y[ix])
        p={k:f[k].astype('f4') for k in KEYS}
    st=stats(tr); log(f'FOLD {label} train={len(tr)} cal={n} eval={len(ev)}')
    p['Regional']=regional(DEVICES[0],st,tr,W[ix],H[ix],alpha=1.2,sigma=.65,local=.7,days=90)
    p['Narrow']=regional(DEVICES[0],st,tr,W[ix],H[ix],alpha=1.2,sigma=.4,local=.85,days=90)
    for name,weight in [('History',1.),('WindHistory',2.)]:
        t=time.perf_counter(); p[name]=atlas_kernel(DEVICES[0],st,tr,LW[ix],H[ix],wind=weight); log(f'FOLD {label} {name} DONE {time.perf_counter()-t:.2f}s')
    assert all(v.shape==(len(ix),70,35,4) and np.isfinite(v).all() for v in p.values())
    if os.environ.get('ATLAS_CACHE','1')=='1':
        np.savez(OUT/f'fold{label}_atlas_predictions.npz',times=TT[ix],n_cal=n,History=p['History'],WindHistory=p['WindHistory'])
    pc={k:v[:n].copy() for k,v in p.items()}; pe={k:v[n:] for k,v in p.items()}; scores={}
    for m in ['Reference',*methods]:
        par=candidate_atlas(pc,Y[ca],H[ca],m); pred=apply_atlas(pe,par,H[ev]); sc,rm=score(pred,Y[ev]); scores[m]=sc
        ROWS.append(dict(fold=str(label),model=m,skill=sc,**dict(zip(['RMSE_'+v for v in VARS],rm.tolist()))))
        log(f'FOLD {label} {m:15s} SKILL={sc:.6f} RMSE={rm.round(4).tolist()}')
    pd.DataFrame(ROWS).to_csv(OUT/'validation_metrics.csv',index=False)
    REPRO.append(abs(scores['Reference']-EXPECTED[str(label)])<.0015)
    if label in (1,2,3): CAL.append((pc,Y[ca],H[ca]))
    SCORES[str(label)]=scores

ROWS=[]; CAL=[]; SCORES={}; REPRO=[]
for f,cut in enumerate(CUTS[:2],1): load_preds(f,cut,NAMES)
gains={m:[SCORES[str(i)][m]-SCORES[str(i)]['Reference'] for i in (1,2)] for m in NAMES}
eligible=[m for m,g in gains.items() if min(g)>=0 and np.mean(g)>=.001]
selected=max(eligible,key=lambda m:np.mean(gains[m])) if eligible else 'Reference'
log(f'FROZEN before retrospective checks: selected={selected} gains={gains}')
for label,cut in [(3,CUTS[2]),('April',STRESS)]: load_preds(label,cut,[] if selected=='Reference' else [selected])
audit=SCORES['3'][selected]-SCORES['3']['Reference']; stress=SCORES['April'][selected]-SCORES['April']['Reference']
PASS=bool(not SMOKE and all(REPRO) and selected!='Reference' and audit>=.001 and stress>=-.001)
log(f'GATE repro={all(REPRO)} audit={audit:+.6f} stress={stress:+.6f} pass={PASS}')
cp={k:np.concatenate([p[k] for p,y,h in CAL]) for k in CAL[0][0]}; cy=np.concatenate([y for p,y,h in CAL]); ch=np.concatenate([h for p,y,h in CAL]); params=candidate_atlas(cp,cy,ch,selected)
tr=np.argsort(TT); ev=np.unique(ti)
if SMOKE: tr=tr[:48]
with np.load(CACHE/'final_family_predictions.npz',allow_pickle=False) as f:
    assert np.array_equal(f['times'],TE[ev]); p={k:f[k].astype('f4') for k in KEYS}
st=stats(tr)
p['Regional']=regional(DEVICES[0],st,tr,WT[ev],HT[ev],alpha=1.2,sigma=.65,local=.7,days=90)
p['Narrow']=regional(DEVICES[0],st,tr,WT[ev],HT[ev],alpha=1.2,sigma=.4,local=.85,days=90)
for name,weight in [('History',1.),('WindHistory',2.)]:
    p[name]=atlas_kernel(DEVICES[0],st,tr,LWT[ev],HT[ev],wind=weight); log(f'FINAL {name} DONE')
if os.environ.get('ATLAS_CACHE','1')=='1':
    np.savez(OUT/'final_atlas_predictions.npz',times=TE[ev],History=p['History'],WindHistory=p['WindHistory'])
forecast=apply_atlas(p,params,HT[ev]); positions=np.full(len(TE),-1,dtype=int); positions[ev]=np.arange(len(ev)); assert (positions[ti]>=0).all()
# If the exact submitted v2.7 is present, compare reconstructed baseline before authorizing a new CSV.
ref=Path(os.environ.get('ATLAS_V27_CSV','/kaggle/working/Ultralisk-v2.7/submission_ultralisk_v2_7.csv'))
ref_rms=None
if ref.is_file() and not SMOKE:
    old=pd.read_csv(ref,dtype={'id':str}); assert old.id.equals(TEMPLATE.id)
    recon=apply27(p,params[0],HT[ev])[positions[ti],ri,ci]
    ref_rms=np.sqrt(np.mean((recon-old[VARS].to_numpy())**2,axis=0)).tolist(); log(f'Exact v2.7 CSV baseline RMS difference={ref_rms}')
    PASS=bool(PASS and max(np.array(ref_rms)/NORM)<.002)
else: log('Exact submitted v2.7 CSV not present; reference checked on historical folds only.')
sub=pd.DataFrame(forecast[positions[ti],ri,ci].round(4),columns=VARS); sub.insert(0,'id',TEMPLATE.id)
name='submission_SMOKE_NO_SUBIR.csv' if SMOKE else ('submission_atlas_v3_0.csv' if PASS else 'submission_atlas_v3_0_NO_SUBIR.csv')
for oldname in ('submission_atlas_v3_0.csv','submission_atlas_v3_0_NO_SUBIR.csv','submission_SMOKE_NO_SUBIR.csv'): (OUT/oldname).unlink(missing_ok=True)
dest=OUT/name; tmp=dest.with_suffix('.tmp'); sub.to_csv(tmp,index=False)
check=pd.read_csv(tmp,dtype={'id':str}); assert check.shape==TEMPLATE.shape and check.columns.tolist()==TEMPLATE.columns.tolist() and check.id.equals(TEMPLATE.id) and check.id.is_unique
assert np.isfinite(check[VARS].to_numpy()).all() and check.RH.between(0,100).all(); tmp.replace(dest)
report=dict(model='Atlas-v3.0',smoke=SMOKE,selected=selected,reference_public_score=.49619,reference_reproduced=all(REPRO),reference_csv_rms=ref_rms,dev_gains=gains,audit_gain=audit,stress_gain=stress,recommend_submit=PASS,weights=None if params[2] is None else params[2].tolist(),validation=ROWS,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),rows=len(check),seconds=time.perf_counter()-START)
(OUT/'report.json').write_text(json.dumps(report,indent=2)); print(pd.DataFrame(ROWS).pivot(index='model',columns='fold',values='skill').round(6).to_string()); log(f'CSV={dest} SHA256={report["sha256"]} rows={len(check)}'); log('RECOMMEND_SUBMIT=YES' if PASS else 'RECOMMEND_SUBMIT=NO: conservar v2.7')
for f in (dest,OUT/'report.json',OUT/'validation_metrics.csv',LOG): display(FileLink(os.path.relpath(f)))
