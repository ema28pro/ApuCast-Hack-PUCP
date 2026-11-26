"""Atlas v3.2.5 patches. Requires unmodified Atlas_v3_2.py beside this file."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import argparse, json, math, time, hashlib, copy, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.linalg import cho_factor, cho_solve
from sklearn.utils.extmath import randomized_svd
import Atlas_v3_2 as v
NORM=v.NORM; VARS=v.VARS
ORIGINS={'O1':('2024031700','2024032400','2024042100'),'O2':('2024041400','2024042100','2024052100'),'O3':('2024052100','2024052800','2024070100')}
PREREG='bcb9d59dc0e13859bd7c6a9d7f6ab63fc4eea2f5175f83892b8520926aff5f37'
CONFIG={'Raw':('raw',False,False),'Placebo':('placebo',False,False),'Physics':('physics',False,False),'Recent':('physics',True,False),'Drift':('physics',False,True)}

def check_protocol():
    p=Path(__file__).with_name('Atlas_v3_2_5_preregistro.md')
    if v.sha(p)!=PREREG: raise ValueError('Preregistration mismatch')

def split(T,h,origin):
    cut,cal,end=ORIGINS[origin]; tr=np.flatnonzero(T<cut); ca=np.flatnonzero((T>=cut)&(T<cal)&(h%3==0)); ev=np.flatnonzero((T>=cal)&(T<end)&(h%3==0)); assert len(tr) and len(ca) and len(ev) and max(T[tr])<cut
    return tr,ca,ev

def zhistory(X,t,h,kind):
    if kind!='placebo': return v.history(X,t,h,kind=='physics')
    r=v.history(X,t,h,False); # generic fixed nonlinear basis, no additional observations
    # Scale in fixed physical units, not fitted to heldout data. 29 channels total.
    q=r/np.tile(np.array([25.36,2.0652,2.6776,2.0876],dtype='f4'),3); q[...,1]-=290/2.0652
    ext=[np.tanh(q[...,j]) for j in range(12)]+[np.tanh(q[...,j])*np.tanh(q[...,(j+1)%12]) for j in range(5)]
    return np.concatenate([r,np.stack(ext,axis=-1)],axis=-1).astype('f4')

def wavg(a,w): return np.tensordot(w,a,axes=(0,0))/w.sum()
def climate(a,h,w,day,drift):
    # day relative to last training timestamp, in units of 30 days.
    base=[]; slopes=[]
    for hour in range(24):
        ix=np.flatnonzero(h==hour); ix=ix if len(ix) else np.arange(len(h)); wi=w[ix]; d=day[ix]; m=wavg(a[ix],wi); dm=np.average(d,weights=wi)
        if drift:
            b=np.tensordot(wi*(d-dm),a[ix],axes=(0,0))/wi.sum()
            b=b/(np.average((d-dm)**2,weights=wi)+.25)
        else:b=np.zeros_like(m)
        base.append(m-b*dm);slopes.append(b)
    return np.asarray(base,dtype='f4'),np.asarray(slopes,dtype='f4')

def dated(T,ref): return (pd.to_datetime(T,format='%Y%m%d%H')-pd.Timestamp(ref)).total_seconds().to_numpy()/86400/30

def seasonal(cl,sl,h,d): return cl[h]+sl[h]*np.clip(d,-100,34/30).reshape((-1,)+(1,)*(cl.ndim-1))

def fit_anchor(X,t,h,Y,T,tr,s,kind='physics',recent=False,drift=False):
    if not recent and not drift and kind!='placebo':
        st=v.fit_anchor(X,t,h,Y,tr,s,kind=='physics');st['legacy']=np.array(1);return st
    z=zhistory(X[tr],t[tr],h[tr],kind); hh=h[tr];tt=t[tr]; n=len(tr);ref=str(pd.to_datetime(T[tr[-1]],format='%Y%m%d%H')); d=dated(T[tr],ref); w=np.exp2(d) if recent else np.ones(n);w=w/w.mean();wf=w.astype('f4')
    cl,sl=climate(z,hh,wf,d,drift);z-=seasonal(cl,sl,hh,d).astype('f4'); sd=np.maximum(np.sqrt(np.average(np.mean(z*z,axis=(1,2)),axis=0,weights=w)),1e-4);z=np.clip(z/sd.astype('f4'),-12,12)
    flat=z.reshape(n,-1);pm=wavg(flat,wf); centered=flat-pm; _,_,pc=randomized_svd(centered*np.sqrt(wf[:,None]),n_components=min(24,n-1),random_state=17); g=centered@pc.T;gs=np.maximum(np.sqrt(np.average(g*g,axis=0,weights=w)),1e-4);g/=gs
    yc,ys=climate(Y[tr],hh,wf,d,drift); yy=((Y[tr]-seasonal(yc,ys,hh,d))/NORM).reshape(n,-1,4)
    lat,lon=np.meshgrid(s['wrf_lats'],s['wrf_lons'],indexing='ij');groups=np.abs(lat.ravel()[:,None]-s['gfs_lats']).argmin(1)*13+np.abs(lon.ravel()[:,None]-s['gfs_lons']).argmin(1)
    st=dict(legacy=np.array(0),kind=np.array(kind),ref=np.array(ref),cl=cl,sl=sl,sd=sd,pca_mean=pm,pca_components=pc,pca_sd=gs,yc=yc,ys=ys,groups=groups)
    for gr in np.unique(groups):
        a=v.design(z,g,tt,gr,kind!='raw');mu=np.average(a,axis=0,weights=w);ds=np.maximum(np.sqrt(np.average((a-mu)**2,axis=0,weights=w)),.05);a=np.column_stack([np.ones(n),(a-mu)/ds]);pix=np.flatnonzero(groups==gr); gram=a.T@(w[:,None]*a);reg=np.full(len(gram),.08*w.sum());reg[0]=1e-6;gram.flat[::len(gram)+1]+=reg;beta=cho_solve(cho_factor(gram),a.T@(w[:,None]*yy[:,pix].reshape(n,-1)))
        for k,x in [('mu',mu),('sd',ds),('beta',beta)]:st[f'{k}_{gr}']=x
    return st

def predict_anchor(st,X,t,h,T):
    if int(st['legacy']):return v.anchor_predict(st,X,t,h)
    kind=str(st['kind']);d=dated(T,str(st['ref']));z=np.clip((zhistory(X,t,h,kind)-seasonal(st['cl'],st['sl'],h,d))/st['sd'],-12,12).astype('f4');g=((z.reshape(len(z),-1)-st['pca_mean'])@st['pca_components'].T)/st['pca_sd'];groups=st['groups'];p=np.empty((len(z),2450,4),dtype='f4')
    for gr in np.unique(groups):
        a=v.design(z,g,t,gr,kind!='raw');a=np.column_stack([np.ones(len(a)),(a-st[f'mu_{gr}'])/st[f'sd_{gr}']]);pix=np.flatnonzero(groups==gr);p[:,pix]=(a@st[f'beta_{gr}']).reshape(len(a),len(pix),4)
    return v.clip(p.reshape(-1,70,35,4)*NORM+seasonal(st['yc'],st['ys'],h,d))

def optimal(a,b,y):
    d=(b.astype('f8')-a)/NORM;r=(y.astype('f8')-a)/NORM;energy=np.sum(d*d,axis=(0,1,2));return np.clip(np.sum(d*r,axis=(0,1,2))/np.maximum(1.25*energy,1e-12),0,.5)

def choose_weights(a,b,y,dates,variable):
    grid=np.array([0,.5,.85,1.]);days=np.unique(dates); losses=np.zeros((4,4)); counts=0
    for day in days[1:]:
        tr=dates<day;ev=dates==day;w=optimal(a[tr],b[tr],y[tr]);
        for i,f in enumerate(grid): losses[i]+=v.sums(v.blend(a[ev],b[ev],w*f),y[ev])[0]
        counts+=int(ev.sum())*2450
    if len(days)<4: factors=np.full(4,.5);common=.5
    else:
        rm=np.sqrt(losses/counts)/NORM;common=float(grid[np.argmin(rm.mean(1))]);factors=.5*grid[np.argmin(rm,axis=0)]+.5*common if variable else np.full(4,common)
    return optimal(a,b,y)*factors,dict(common=common,factors=factors.tolist(),prequential_days=len(days))

def cache_meta(data_hash,tr,T,both,origin,mode):
    return dict(origin=origin,mode=mode,cut=ORIGINS[origin][0],train_max=str(max(T[tr])),train_times=T[tr].tolist(),query_times=T[both].tolist(),data=data_hash,ultra_sha256=hashlib.sha256(v.ULTRA_SOURCE.encode()).hexdigest(),patch_sha256=v.sha(__file__),epochs=[25,36] if mode=='full' else None)

def reference(u,data_hash,tr,ca,ev,T,h,Y,origin,mode,out):
    both=np.r_[ca,ev];nc=len(ca);meta=cache_meta(data_hash,tr,T,both,origin,mode);path=out/'reference_cache.npz';start=time.perf_counter();hit=path.exists()
    if hit:
        z=v.readnp(path);stored=json.loads(str(z.pop('metadata')));assert stored==meta and stored['train_max']<stored['cut'] and np.array_equal(z.pop('times'),T[both]);p=z
    else:
        st=u['stats'](tr);d=u['DEVICES'][0]
        if mode=='proxy':p={'Narrow':u['regional'](d,st,tr,u['W'][both],h[both],sigma=.4,local=.85,days=90)}
        else:
            p=u['run_stage'](tr,u['W'][both],h[both],origin,u['FAMILIES']);p['Regional']=u['regional'](d,st,tr,u['W'][both],h[both],sigma=.65,local=.7,days=90);p['Narrow']=u['regional'](d,st,tr,u['W'][both],h[both],sigma=.4,local=.85,days=90)
        np.savez(path,metadata=json.dumps(meta,sort_keys=True),times=T[both],**p)
    assert set(p)==({'Narrow'} if mode=='proxy' else set(u['KEYS27'])) and all(x.shape==(len(both),70,35,4) and np.isfinite(x).all() for x in p.values())
    def f(ids,inds):
        if mode=='proxy':return v.clip(p['Narrow'][inds]+v.bias(p['Narrow'][ids],Y[ca[ids]],h[ca[ids]])[h[both[inds]]])
        w=u['calibration']({k:x[ids] for k,x in p.items()},Y[ca[ids]],'NarrowSpatial',h[ca[ids]]);return u['apply27']({k:x[inds] for k,x in p.items()},w,h[both[inds]])
    dt=v.daystr(T[ca]);idx=[];pre=[]
    for day in np.unique(dt)[2:]:
        fit=np.flatnonzero(dt<day);val=np.flatnonzero(dt==day);idx.append(val);pre.append(f(fit,val))
    if mode=='full':
        ww,mm=u['calibration']({k:x[:nc] for k,x in p.items()},Y[ca],'NarrowSpatial',h[ca]);np.savez(out/'ultra_calibration.npz',weights=ww,bias=mm)
    return f(np.arange(nc),np.arange(len(both))),np.concatenate(idx),np.concatenate(pre),dict(cache_hit=hit,seconds=time.perf_counter()-start,sha256=v.sha(path),train_max=meta['train_max'],cut=meta['cut'])

def corrected(a,ca,both,h,Y,T,oid):
    nc=len(ca);dates=v.daystr(T[ca]);frozen=v.clip(a+v.bias(a[:nc],Y[ca],h[ca])[h[both]]);pre=[]
    for day in np.unique(dates[oid]):
        tr=dates<day;val=oid[dates[oid]==day];pre.append(v.clip(a[val]+v.bias(a[:nc][tr],Y[ca][tr],h[ca][tr])[h[ca[val]]]))
    return frozen,np.concatenate(pre)

def metrics(pred,y,T,origin,out):
    start=pd.to_datetime(ORIGINS[origin][1],format='%Y%m%d%H');end=pd.to_datetime(ORIGINS[origin][2],format='%Y%m%d%H');days=pd.date_range(start,end-pd.Timedelta(days=1)).strftime('%Y%m%d').to_numpy(dtype='U8');dt=v.daystr(T);ns=np.array([(dt==d).sum()*2450 for d in days]);rows=[];weekly=[];daily=[];allss={}
    for name,p in pred.items():
        ss=np.array([v.sums(p[dt==d],y[dt==d])[0] if n else np.zeros(4) for d,n in zip(days,ns)]);allss[name]=ss
        def row(s,n):return dict(origin=origin,model=name,skill=float(v.skill(s,n)),pixels_hours=int(n),**dict(zip(['RMSE_'+j for j in VARS],np.sqrt(s/n))))
        rows.append(row(ss.sum(0),ns.sum()))
        for i,n in enumerate(ns):
            if n:daily.append(dict(date=days[i],day=i+1,**row(ss[i],n)))
        for i in range(0,len(ns),7):
            n=ns[i:i+7].sum()
            if n:weekly.append(dict(week=i//7+1,**row(ss[i:i+7].sum(0),n)))
    for name,r in [('validation_metrics',rows),('weekly_scores',weekly),('daily_scores',daily)]:pd.DataFrame(r).to_csv(out/f'{name}.csv',index=False)
    np.savez(out/'daily_sse.npz',days=days,counts=ns,**allss);return rows

def pooled_gate(records,candidate,full):
    def gain(items):
        a=sum(x['Reference'].sum(0) for x in items);b=sum(x[candidate].sum(0) for x in items);n=sum(x['counts'].sum() for x in items);return float(v.skill(b,n)-v.skill(a,n))
    r=records['O3'];n=r['counts'];a=r['Reference'];b=r[candidate];good=n>0
    def partial(q):return float(v.skill(b[q].sum(0),n[q].sum())-v.skill(a[q].sum(0),n[q].sum()))
    def wins(x):q=x['counts']>0;return np.sum(v.skill(x[candidate][q],x['counts'][q])>v.skill(x['Reference'][q],x['counts'][q])),q.sum()
    w,nn=map(sum,zip(*(wins(x) for x in records.values())));out=dict(last_gain=gain([r]),late_weeks_gain=partial(slice(14,None)),last14_gain=partial(slice(-14,None)),last_win_fraction=float(wins(r)[0]/wins(r)[1]),pooled_gain=gain(list(records.values())),pooled_win_fraction=float(w/nn),origins={k:gain([x]) for k,x in records.items()});out['leave_one_out']={k:gain([x for j,x in records.items() if j!=k]) for k in records} if len(records)>1 else {}
    for L in (3,7):
        rng=np.random.default_rng(2026+L); B=10000;aa=np.zeros((B,4));bb=aa.copy();nn=np.zeros(B)
        for x in records.values():
            D=len(x['counts']);idx=(rng.integers(0,D-L+1,size=(B,math.ceil(D/L)))[...,None]+np.arange(L)).reshape(B,-1)[:,:D];aa+=x['Reference'][idx].sum(1);bb+=x[candidate][idx].sum(1);nn+=x['counts'][idx].sum(1)
        z=v.skill(bb[nn>0],nn[nn>0])-v.skill(aa[nn>0],nn[nn>0]);out[f'bootstrap{L}']=dict(ci95=np.quantile(z,[.025,.975]).tolist(),se=float(z.std(ddof=1)),replicates=len(z),approx_nonoverlap_blocks=sum(len(x['counts'])//L for x in records.values()))
    out['cpu_signal']=bool(out['last_gain']>0 and out['late_weeks_gain']>0 and out['last_win_fraction']>=.65)
    out['review_eligible']=bool(full and set(records)==set(ORIGINS) and out['cpu_signal'] and out['last14_gain']>=0 and out['pooled_gain']>0 and out['pooled_win_fraction']>=.65 and min(out['leave_one_out'].values())>0 and all(out[f'bootstrap{L}']['ci95'][0]>0 for L in (3,7)))
    return out

class Conditioned(nn.Module):
    def __init__(self,s,mu,sd,width,condition):
        super().__init__();self.net=v.model_new(s,mu,sd,width,True);self.condition=condition;old=self.net.fuse[0];new=nn.Conv2d(old.in_channels+4,old.out_channels,1,bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:old.in_channels].copy_(old.weight)
            if old.bias is not None:new.bias.copy_(old.bias)
        self.net.fuse[0]=new
    def forward(self,x,t,a):
        q=((a-a.new_tensor([70,290,0,0]))/a.new_tensor(NORM.astype('f4'))).permute(0,3,1,2).to(x.dtype);q=q if self.condition else torch.zeros_like(q)
        def inject(module,args):return (torch.cat([args[0],q.to(args[0].dtype)],1),)
        handle=self.net.fuse[0].register_forward_pre_hook(inject)
        try:return self.net(x,t)
        finally:handle.remove()

def train_conditioned(X,t,Y,ix,anchor,s,kind,condition,seed,epochs,device,width,path):
    start=time.perf_counter();mu=np.tile(X[ix,:15].mean((0,2,3)),3);sd=np.tile(np.maximum(X[ix,:15].std((0,2,3)),1e-4),3)
    with v.LOCK:torch.manual_seed(seed);model=Conditioned(s,mu,sd,width,condition)
    model=model.to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);rng=np.random.default_rng(seed);batch=32 if width==48 else 16
    xa=torch.as_tensor((X[ix]-mu[None,:,None,None])/sd[None,:,None,None],device=device);ta=torch.as_tensor(t[ix],device=device);aa=torch.as_tensor(anchor,device=device);yy=torch.as_tensor((Y[ix]-anchor)/NORM.astype('f4'),device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=math.ceil(len(ix)/batch)*epochs,eta_min=2e-5);amp=device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp);updates=0;times=[]
    for ep in range(epochs):
        es=time.perf_counter();model.train();order=rng.permutation(len(ix))
        for j in range(0,len(ix),batch):
            k=torch.as_tensor(order[j:j+batch],device=device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):p=model(xa[k],ta[k],aa[k])
            e=p.float()-yy[k];m=e.square().mean((0,1,2));lv=.6*(m+1e-6).sqrt()+.4*m;loss=lv.mean()
            if kind=='wind':loss=.35*lv[2]+.65*lv[3]+.01*(e[...,2:].diff(dim=1).square().mean()+e[...,2:].diff(dim=2).square().mean())
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);old=scaler.get_scale();scaler.step(opt);scaler.update()
            if scaler.get_scale()>=old:
                sched.step();updates+=1
                with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-min(.995,(1+updates)/(10+updates)))
        if amp:torch.cuda.synchronize(device)
        times.append(time.perf_counter()-es);print(kind,condition,seed,ep+1,times[-1],flush=True)
    torch.save(dict(state=ema.cpu().state_dict(),mu=torch.from_numpy(mu),sd=torch.from_numpy(sd),width=width,kind=kind,condition=condition,seed=seed,epochs=epochs),path)
    return dict(seed=seed,kind=kind,condition=condition,seconds=time.perf_counter()-start,epoch_seconds=times,device=str(device),updates=updates)

def predict_net(path,X,t,s,anchor,device):
    ck=torch.load(path,map_location='cpu',weights_only=True);mu=ck['mu'].numpy();sd=ck['sd'].numpy();model=Conditioned(s,mu,sd,ck['width'],ck['condition']);model.load_state_dict(ck['state']);model=model.to(device).eval();p=[]
    with torch.inference_mode():
        for j in range(0,len(X),32):
            sl=slice(j,j+32);p.append(model(torch.as_tensor((X[sl]-mu[None,:,None,None])/sd[None,:,None,None],device=device),torch.as_tensor(t[sl],device=device),torch.as_tensor(anchor[sl],device=device)).cpu().numpy())
    d=np.concatenate(p)*NORM
    if ck['kind']=='wind':d[...,:2]=0
    return v.clip(anchor+d)

def load_or_fit(out,name,X,t,h,Y,T,tr,s,dh,origin):
    path=out/f'anchor_{name}.npz'; meta=dict(config=list(CONFIG[name]),data=dh,train_times=T[tr].tolist(),cut=ORIGINS[origin][0],code=v.sha(__file__),legacy_code=v.sha(v.__file__))
    if path.exists():
        st=v.readnp(path)
        if 'fit_metadata' not in st or json.loads(str(st['fit_metadata']))!=meta:raise ValueError(f'Anchor cache provenance mismatch: {path}')
        return st
    st=fit_anchor(X,t,h,Y,T,tr,s,*CONFIG[name]);st['fit_metadata']=np.array(json.dumps(meta,sort_keys=True));np.savez(path,**st);return st

def run_origin(args,data,dh,u,X,t,h,Y,T,s,origin):
    start=time.perf_counter();out=Path(args.out)/origin;out.mkdir(parents=True,exist_ok=True);tr,ca,ev=split(T,h,origin);both=np.r_[ca,ev];nc=len(ca);mode='proxy' if args.mode=='proxy' else 'full';bp,oid,boof,meta=reference(u,dh,tr,ca,ev,T,h,Y,origin,mode,out);pred={'Reference':bp};policies={};pre={};stphys=None
    for name,(kind,rec,drift) in CONFIG.items():
        st=load_or_fit(out,name,X,t,h,Y,T,tr,s,dh,origin);p=predict_anchor(st,X[both],t[both],h[both],T[both]);pred[name]=p;pre[name]=p[oid];print(origin,name,v.score(p[nc:],Y[ev]),flush=True)
        if name=='Physics':stphys=st;pred['Bias'],pre['Bias']=corrected(p,ca,both,h,Y,T,oid);np.savez(out/'anchor_bias.npz',bias=v.bias(p[:nc],Y[ca],h[ca]))
    timings=[];cross_seconds=None
    if args.neural and origin=='O3':
        cs=time.perf_counter();ix,oo,man=v.crossfit(X,t,h,Y,T,tr,s,True);cross_seconds=time.perf_counter()-cs;pd.DataFrame(man).to_csv(out/'crossfit_manifest.csv',index=False);inside=predict_anchor(stphys,X[ix],t[ix],h[ix],T[ix]);dev=[torch.device('cuda:0'),torch.device('cuda:1')];epochs=args.epochs
        from concurrent.futures import ThreadPoolExecutor
        for label,kind,cond,aa in [('InNoInput','physics',False,inside),('InInput','physics',True,inside),('OOFNoInput','physics',False,oo),('OOFInput','physics',True,oo),('WindInput','wind',True,oo)]:
            paths=[out/f'{label}_{seed}.pt' for seed in v.SEEDS]
            with ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(train_conditioned,X,t,Y,ix,aa,s,kind,cond,seed,epochs,dev[j],48,paths[j]) for j,seed in enumerate(v.SEEDS)];rr=[f.result() for f in fs]
            pp=[predict_net(p,X[both],t[both],s,pred['Physics'],dev[0]) for p in paths];pred[label]=v.clip(np.mean(pp,axis=0));pre[label]=pred[label][oid]
            for j,r in enumerate(rr):
                replay=predict_net(paths[j],X[both][:32],t[both][:32],s,pred['Physics'][:32],dev[0]);r['replay_max_abs']=float(np.max(np.abs(replay-pp[j][:32])));assert r['replay_max_abs']<=1e-4
                r['seed_skill']=v.score(pp[j][nc:],Y[ev]);timings.append(dict(variant=label,**r))
    for name in list(pre):
        for variable in ([False,True] if name=='Bias' else [True]):
            label='Mix'+name+('Var' if variable else 'Common');w,info=choose_weights(boof,pre[name],Y[ca[oid]],v.daystr(T[ca[oid]]),variable)
            if name=='WindInput':w[:2]=0
            pred[label]=v.blend(bp,pred[name],w);policies[label]=dict(weights=w.tolist(),**info)
    if args.neural and origin=='O3':
        # Stage 1 bias blend uses only previous prequential days for each day.
        dates=v.daystr(T[ca[oid]]);use=[];pstage=[]
        for day in np.unique(dates)[1:]:
            q=dates<day;r=dates==day;w0,_=choose_weights(boof[q],pre['Bias'][q],Y[ca[oid[q]]],dates[q],True);pstage.append(v.blend(boof[r],pre['Bias'][r],w0));use.extend(np.flatnonzero(r))
        use=np.asarray(use);pstage=np.concatenate(pstage)
        for label in ('InNoInput','InInput','OOFNoInput','OOFInput','WindInput'):
            w,info=choose_weights(pstage,pre[label][use],Y[ca[oid[use]]],dates[use],True)
            if label=='WindInput':w[:2]=0
            pred['MixBiasPlus'+label]=v.blend(pred['MixBiasVar'],pred[label],w);policies['MixBiasPlus'+label]=dict(weights=w.tolist(),base='MixBiasVar',**info)
    rows=metrics({k:p[nc:] for k,p in pred.items()},Y[ev],T[ev],origin,out);corr=[]
    for name in pre:
        for scope,a,b,y in [('calibration',boof,pre[name],Y[ca[oid]]),('evaluation',bp[nc:],pred[name][nc:],Y[ev])]:
            e=a.astype('f8')-y;f=b.astype('f8')-y
            for j,var in enumerate(VARS):corr.append(dict(origin=origin,model=name,scope=scope,variable=var,pearson=float(np.corrcoef(e[...,j].ravel(),f[...,j].ravel())[0,1])))
    pd.DataFrame(corr).to_csv(out/'correlations.csv',index=False);np.savez(out/'static.npz',**s);np.savez(out/'validation_predictions.npz',times=T[both],n_cal=nc,target=Y[both],**pred);report=dict(origin=origin,mode=args.mode,reference='Narrow proxy' if mode=='proxy' else 'reconstructed v2.7',cache=meta,counts=[len(tr),len(ca),len(ev)],policies=policies,neural_timings=timings,crossfit_seconds=cross_seconds,wall_seconds=time.perf_counter()-start,code_sha256=v.sha(__file__),data=dh,seed_list=v.SEEDS);v.dump(out/'report.json',report);print(origin,pd.DataFrame(rows)[['model','skill']].to_string(index=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['proxy','full','benchmark','infer'],default='proxy');p.add_argument('--data',default='/kaggle/input');p.add_argument('--out',required=True);p.add_argument('--origin',choices=['all',*ORIGINS],default='all');p.add_argument('--neural',action='store_true');p.add_argument('--epochs',type=int,default=48);p.add_argument('--cpu-report');p.add_argument('--benchmark-report');p.add_argument('--bundle');p.add_argument('--input');p.add_argument('--fit-only',action='store_true');p.add_argument('--component',default='Physics',choices=list(CONFIG)+['Bias','InNoInput','InInput','OOFNoInput','OOFInput','WindInput']);args=p.parse_args();check_protocol();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.mode=='infer':
        src=Path(args.bundle);is_net=args.component in ['InNoInput','InInput','OOFNoInput','OOFInput','WindInput'];anchor_name='Physics' if args.component=='Bias' or is_net else args.component;manifest=json.loads((src.parent/'manifest.json').read_text());assert manifest['code'][Path(__file__).name]==v.sha(__file__) and manifest['code']['Atlas_v3_2.py']==v.sha(v.__file__)
        for name in ['static.npz',f'anchor_{anchor_name}.npz']+(['anchor_bias.npz'] if args.component=='Bias' else [])+([f'{args.component}_{seed}.pt' for seed in v.SEEDS] if is_net else []):assert v.sha(src/name)==manifest['files'][str(Path(src.name)/name)]
        s=v.readnp(src/'static.npz')
        with np.load(args.input,allow_pickle=False) as f:X0=f['X'];T=f['times'].astype(str)
        X,t,h=v.A['prepare'](X0,T,s);st=v.readnp(src/f'anchor_{anchor_name}.npz');q=predict_anchor(st,X,t,h,T)
        if args.component=='Bias':q=v.clip(q+v.readnp(src/'anchor_bias.npz')['bias'][h])
        if is_net:
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');q=v.clip(np.mean([predict_net(src/f'{args.component}_{seed}.pt',X,t,s,q,device) for seed in v.SEEDS],axis=0))
        np.savez(out/'atlas_predictions.npz',times=T,prediction=q);return
    if args.mode=='proxy' and args.neural:raise ValueError('Neural training requires GPU phase')
    if args.mode=='benchmark':args.neural=True;assert args.epochs==2 and args.origin=='O3','Benchmark requires --epochs 2 --origin O3'
    if args.neural:
        if not args.cpu_report or not json.loads(Path(args.cpu_report).read_text())['gate']['cpu_signal']:raise ValueError('CPU signal required')
        if args.mode=='full':
            if not args.benchmark_report:raise ValueError('Benchmark required')
            br=json.loads(Path(args.benchmark_report).read_text());assert br['mode']=='benchmark' and br['epochs']==2 and br['reports'][0]['neural_timings'] and br['preregister']==PREREG and br['reports'][0]['code_sha256']==v.sha(__file__)
    torch.use_deterministic_algorithms(True,warn_only=True);data=v.data_root(args.data);dh=v.fingerprint(data);train=v.readnp(data/'train.npz');s=v.readnp(data/'static.npz');T=train['times'].astype(str);Y=train['Y'];X,t,h=v.A['prepare'](train['X'],T,s)
    origins=list(ORIGINS) if args.origin=='all' else [args.origin]
    if args.fit_only:
        for origin in origins:
            tr,_,_=split(T,h,origin);dest=out/origin;dest.mkdir(parents=True,exist_ok=True);load_or_fit(dest,args.component,X,t,h,Y,T,tr,s,dh,origin)
        return
    u=v.init_ultra(data,out,'smoke' if args.mode=='proxy' else 'full');reports=[]
    for origin in origins:reports.append(run_origin(args,data,dh,u,X,t,h,Y,T,s,origin))
    records={k:v.readnp(out/k/'daily_sse.npz') for k in origins};gate=pooled_gate(records,'MixBiasVar',args.mode=='full' and (not args.neural or args.epochs>=48)) if 'O3' in records else None
    summary=dict(mode=args.mode,epochs=args.epochs,reports=reports,gate=gate,scope='retrospective; no automatic submission',preregister=PREREG,reference_scope='Narrow only' if args.mode=='proxy' else 'reconstructed Ultralisk 2.7')
    if args.neural and args.mode=='benchmark':
        rr=reports[0]['neural_timings'];summary['projected_atlas_epoch_seconds_48']=sum(max(x['epoch_seconds'][-1] for x in rr if x['variant']==k) for k in set(x['variant'] for x in rr))*48;summary['benchmark_note']='Training-only estimate for O3; excludes anchor, inference and v2.7 reconstruction.'
    v.dump(out/'report.json',summary);v.dump(out/'manifest.json',dict(code={str(Path(__file__).name):v.sha(__file__),'Atlas_v3_2.py':v.sha(v.__file__)},preregister=PREREG,data=dh,origins=ORIGINS,seeds=v.SEEDS,torch=str(torch.__version__),cuda=torch.version.cuda,python=platform.python_version(),replay_atol=1e-4,files={str(p.relative_to(out)):v.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}));print('GATE',gate,flush=True)
if __name__=='__main__':main()
