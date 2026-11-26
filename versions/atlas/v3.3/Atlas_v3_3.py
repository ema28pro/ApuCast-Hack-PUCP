"""Atlas v3.3 AdaptiveFlow: retrospective experiments, no submission."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn,idctn
import Atlas_v3_2_5 as a
v=a.v
m325=a
CANDIDATES=['PhysicsNewBias','RecentBias','DriftBias','MultiScale','AdaptiveFlow','WindFlow','Taper']
COUNTS={'O1':[666,32,208],'O2':[1215,56,206],'O3':[1999,56,256]}
EXPECTED={'Atlas_v3_2.py':'cd60f877d245cd2955187788e562b6fe8fcb4c448829514cbdf376b81455829e','Atlas_v3_2_5.py':'4411658eb868a353f72815ece14101b4c7fc135d8e7d20973e865ec98971f55e'}
OFFICIAL_HASHES={'sample_submission.csv': '507d20c815b09f253b43696dc611312355672f1552d2b313e940b32f7f0ca6a7', 'static.npz': '0e181ff372a9a33fdbbcede81da6ddd2da8208f15b94594d78d3f10464852165', 'test.npz': '97894492e5caae72e45b09f0d4d8bd87e03c12e5029f33d276b353870aa5c0bd', 'train.npz': 'f5120b7a7cf5661e8e4143a726f2d22d13905d11d878afb60c8ff5cd02444844'}
FULL_HISTORY={'O1': [0.5027231229629487, 0.5034352685502492], 'O2': [0.4739757924262317, 0.476131105418833], 'O3': [0.4309236143627557, 0.4388617738091176]}
FULL_AUDIT_SHA="724ca9c3b31bd82852e2019e3db4dc3d2439d0188092c9868f7a1ba8ecfb17c7"
# Lock selection on O1/O2: positive in both, highest pooled score. O3 is audit only.
# Use identical incumbent weights for all variants; no evaluation labels in fit.
def fit_new(X,t,h,Y,T,tr,s,kind='physics',recent=False,drift=False):
    """New v3.2.5 anchor route, without legacy dispatch."""
    z=m325.zhistory(X[tr],t[tr],h[tr],kind); hh=h[tr];tt=t[tr]; n=len(tr);ref=str(pd.to_datetime(T[tr[-1]],format='%Y%m%d%H')); d=m325.dated(T[tr],ref); w=np.exp2(d) if recent else np.ones(n);w=w/w.mean();wf=w.astype('f4')
    cl,sl=m325.climate(z,hh,wf,d,drift);z-=m325.seasonal(cl,sl,hh,d).astype('f4'); sd=np.maximum(np.sqrt(np.average(np.mean(z*z,axis=(1,2)),axis=0,weights=w)),1e-4);z=np.clip(z/sd.astype('f4'),-12,12)
    flat=z.reshape(n,-1);pm=m325.wavg(flat,wf); centered=flat-pm; _,_,pc=m325.randomized_svd(centered*np.sqrt(wf[:,None]),n_components=min(24,n-1),random_state=17); g=centered@pc.T;gs=np.maximum(np.sqrt(np.average(g*g,axis=0,weights=w)),1e-4);g/=gs
    yc,ys=m325.climate(Y[tr],hh,wf,d,drift); yy=((Y[tr]-m325.seasonal(yc,ys,hh,d))/v.NORM).reshape(n,-1,4)
    lat,lon=np.meshgrid(s['wrf_lats'],s['wrf_lons'],indexing='ij');groups=np.abs(lat.ravel()[:,None]-s['gfs_lats']).argmin(1)*13+np.abs(lon.ravel()[:,None]-s['gfs_lons']).argmin(1)
    st=dict(legacy=np.array(0),kind=np.array(kind),ref=np.array(ref),cl=cl,sl=sl,sd=sd,pca_mean=pm,pca_components=pc,pca_sd=gs,yc=yc,ys=ys,groups=groups)
    for gr in np.unique(groups):
        a=v.design(z,g,tt,gr,kind!='raw');mu=np.average(a,axis=0,weights=w);ds=np.maximum(np.sqrt(np.average((a-mu)**2,axis=0,weights=w)),.05);a=np.column_stack([np.ones(n),(a-mu)/ds]);pix=np.flatnonzero(groups==gr); gram=a.T@(w[:,None]*a);reg=np.full(len(gram),.08*w.sum());reg[0]=1e-6;gram.flat[::len(gram)+1]+=reg;beta=m325.cho_solve(m325.cho_factor(gram),a.T@(w[:,None]*yy[:,pix].reshape(n,-1)))
        for k,x in [('mu',mu),('sd',ds),('beta',beta)]:st[f'{k}_{gr}']=x
    return st

def spectral_smooth(x):
    z=dctn(x,axes=(0,1),norm='ortho');z[6:]=0;z[:,3:]=0
    return idctn(z,axes=(0,1),norm='ortho').astype('f4')
def features(p):
    u,w=p[...,2],p[...,3];return np.stack([u,w,np.hypot(u,w)],axis=-1)
def fit_correction(p,y,h,variant):
    assert len(p)==len(y)==len(h)>0
    st={'bias':v.bias(p,y,h)}
    if variant=='MultiScale':st['bias']=np.stack([.6*b+.4*spectral_smooth(b) for b in st['bias']])
    if variant in ('AdaptiveFlow','WindFlow'):
        f=features(p);mu=f.mean(0);sd=np.maximum(f.std(0),.5);z=np.clip((f-mu)/sd,-3,3).astype('f8')
        e=(y-p-st['bias'][h])/v.NORM;e=e-e.mean(0)
        gram=np.einsum('nhwi,nhwj->hwij',z,z)/len(z)+.75*np.eye(3)
        rhs=np.einsum('nhwi,nhwv->hwiv',z,e)/len(z)
        beta=gaussian_filter(np.linalg.solve(gram,rhs).astype('f4'),sigma=(3,3,0,0))
        if variant=='WindFlow':beta[...,:2]=0
        st.update(mu=mu,sd=sd,beta=beta)
    return st
def apply_correction(st,p,h,T,last,variant):
    b=st['bias'][h]
    if variant=='Taper':b=b*np.exp2(-np.maximum(a.dated(T,last)*30,0)/21.)[:,None,None,None]
    if 'beta' in st:
        z=np.clip((features(p)-st['mu'])/st['sd'],-3,3)
        extra=.35*np.einsum('nhwi,hwiv->nhwv',z,st['beta'])*v.NORM
        b=b+np.clip(extra,-.3*v.NORM,.3*v.NORM)
    return v.clip(p+b)
def fit_apply(p,y,h,T,nc,variant):
    assert len(p)==len(h)==len(T) and len(y)==nc and T[nc-1]<T[nc]
    st=fit_correction(p[:nc],y,h[:nc],variant);last=pd.to_datetime(T[nc-1],format='%Y%m%d%H')
    q=apply_correction(st,p,h,T,last,variant)
    assert np.max(np.abs(apply_correction(st,p[nc:nc+1],h[nc:nc+1],T[nc:nc+1],last,variant)-q[nc:nc+1]))<=1e-5
    return q
def run_origin(args,u,dh,X,t,h,Y,T,s,o):
    dest=Path(args.out)/o;dest.mkdir(parents=True,exist_ok=True)
    tr,ca,ev=a.split(T,h,o);both=np.r_[ca,ev];nc=len(ca)
    assert [len(tr),nc,len(ev)]==COUNTS[o] and max(T[tr])<min(T[ca])<=max(T[ca])<min(T[ev])
    refdir=Path(args.cache)/o if args.cache else dest;refdir.mkdir(parents=True,exist_ok=True)
    bp,oid,boof,cache=a.reference(u,dh,tr,ca,ev,T,h,Y,o,args.mode,refdir)
    def anchor(name,kind,rec=False,drift=False):
        st=fit_new(X,t,h,Y,T,tr,s,kind,rec,drift) if name=='PhysicsNew' else a.load_or_fit(refdir,name,X,t,h,Y,T,tr,s,dh,o)
        return a.predict_anchor(st,X[both],t[both],h[both],T[both])
    phys=anchor('Physics','physics');old,pre=a.corrected(phys,ca,both,h,Y,T,oid)
    w,info=a.choose_weights(boof,pre,Y[ca[oid]],v.daystr(T[ca[oid]]),True)
    pred={'Ultralisk':bp,'PhysicsLegacy':phys,'Bias325':old,'Reference':v.blend(bp,old,w)}
    if (refdir/'validation_metrics.csv').exists() and refdir!=dest:
        m=pd.read_csv(refdir/'validation_metrics.csv').set_index('model');assert abs(v.score(pred['Reference'][nc:],Y[ev])-m.loc['MixBiasVar','skill'])<1e-6
    for name,kind,rec,drift in [('PhysicsNew','physics',False,False),('Placebo','placebo',False,False),('Recent','physics',True,False),('Drift','physics',False,True)]:
        p=anchor(name,kind,rec,drift);pred[name]=p
        if name!='Placebo':pred[name+'Bias']=fit_apply(p,Y[ca],h[both],T[both],nc,'Bias')
    for name in ['MultiScale','AdaptiveFlow','WindFlow','Taper']:pred[name]=fit_apply(phys,Y[ca],h[both],T[both],nc,name)
    for name in CANDIDATES:pred['Mix'+name]=v.blend(bp,pred[name],w)
    for p in pred.values():assert p.shape==Y[both].shape and np.isfinite(p).all() and np.all((p[...,0]>=0)&(p[...,0]<=100))
    rows=a.metrics({k:p[nc:] for k,p in pred.items()},Y[ev],T[ev],o,dest);base=next(r['skill'] for r in rows if r['model']=='Reference')
    if args.mode=='full':
        current=[v.score(bp[nc:],Y[ev]),base]
        if not np.allclose(current,FULL_HISTORY[o],rtol=0,atol=.001):raise ValueError(f'Historical baseline drift {o}: {current} vs {FULL_HISTORY[o]}')
    print(o,'baseline',base,flush=True)
    for r in rows:
        if r['model'].startswith('Mix'):print(r['model'],r['skill']-base,flush=True)
    report=dict(origin=o,mode=args.mode,reference='MixBiasVar / '+('Narrow proxy' if args.mode=='proxy' else 'reconstructed Ultralisk v2.7'),counts=COUNTS[o],cache=cache,weights=w.tolist(),policy=info,data=dh,code_sha256=v.sha(__file__))
    v.dump(dest/'report.json',report);return report
def gain(rs,name):
    n=sum(x['counts'].sum() for x in rs);b=sum(x['Mix'+name].sum(0) for x in rs);r=sum(x['Reference'].sum(0) for x in rs);return float(v.skill(b,n)-v.skill(r,n))
def select_dev(records):
    rs=[records[o] for o in ('O1','O2')];ok=[n for n in CANDIDATES if min(gain([r],n) for r in rs)>0];return max(ok,key=lambda n:gain(rs,n)) if ok else None
def tests():
    rng=np.random.default_rng(33);p=rng.normal(size=(8,70,35,4)).astype('f4');p[...,0]+=50;p[...,1]+=280;y=p.copy();h=np.arange(8)*3;T=np.array([f'20240501{x:02}' for x in h]);last=pd.Timestamp('2024-05-01')
    for n in ['Bias','MultiScale','AdaptiveFlow','WindFlow','Taper']:assert np.array_equal(p,apply_correction(fit_correction(p,y,h,n),p,h,T,last,n)),n
    assert np.max(np.abs(spectral_smooth(np.ones((70,35,4),'f4'))-1))<1e-5
    assert np.all(fit_correction(p,y+1,h,'WindFlow')['beta'][...,:2]==0)
    fit_apply(np.concatenate([p,p]),y,np.tile(h,2),np.r_[T,np.char.replace(T,'20240501','20240502')],8,'AdaptiveFlow')
    print('TESTS OK: zero residual, constant DCT, wind isolation, replay',flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['proxy','full'],default='proxy');p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--cache');p.add_argument('--locked');args=p.parse_args()
    start=time.perf_counter();a.check_protocol();tests();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for name,digest in EXPECTED.items():assert v.sha(Path(__file__).with_name(name))==digest
    if args.mode=='full':
        import torch
        assert args.locked=='AdaptiveFlow', 'Full follow-up is locked to exploratory AdaptiveFlow'
        assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2));torch.use_deterministic_algorithms(True,warn_only=True)
    data=v.data_root(args.data);dh=v.fingerprint(data);train=v.readnp(data/'train.npz');s=v.readnp(data/'static.npz');T=train['times'].astype(str);Y=train['Y'];X,t,h=v.A['prepare'](train['X'],T,s)
    assert dh==OFFICIAL_HASHES,'Official data hashes differ'
    u=v.init_ultra(data,out,'smoke' if args.mode=='proxy' else 'full');records={};reports=[]
    for o in ('O1','O2','O3'):
        if o=='O3':
            selected=select_dev(records) if args.mode=='proxy' else args.locked
            v.dump(out/'selection_before_O3.json',dict(selected=selected,mode=args.mode,dev_gains={n:{k:gain([r],n) for k,r in records.items()} for n in CANDIDATES}));print('LOCKED BEFORE O3:',selected,flush=True)
        reports.append(run_origin(args,u,dh,X,t,h,Y,T,s,o));records[o]=v.readnp(out/o/'daily_sse.npz')
    gate=a.pooled_gate(records,'Mix'+selected,args.mode=='full') if selected else None
    if gate:
        gate['statistical_gate_passed']=bool(gate['review_eligible'] and min(gate['origins'].values())>0)
        # Full follow-up chosen after proxy O3, hence not independent confirmation.
        gate['review_eligible']=False
    exploratory={n:a.pooled_gate(records,'Mix'+n,False) for n in CANDIDATES}
    summary=dict(version='3.3',mode=args.mode,selected=selected,gate=gate,exploratory_gates=exploratory,selection_provenance=('CPU O1/O2 selection before current O3' if args.mode=='proxy' else 'AdaptiveFlow locked before full run, selected after retrospective proxy O3'),historical_audit_sha256=FULL_AUDIT_SHA,source_sha256=v.sha(__file__),scope='retrospective; no submission; Reference=MixBiasVar',data=dh,reports=reports,seconds=time.perf_counter()-start)
    v.dump(out/'report.json',summary);frames=[]
    def add(section,df):df=df.copy();df.insert(0,'section',section);frames.append(df)
    add('metadata',pd.DataFrame([{'field':k,'value':json.dumps(x,sort_keys=True)} for k,x in summary.items()]))
    for o,r in records.items():
        for section,file in [('global','validation_metrics.csv'),('weekly','weekly_scores.csv'),('daily','daily_scores.csv')]:add(section,pd.read_csv(out/o/file))
        for name in sorted(set(r)-{'days','counts'}):
            d=pd.DataFrame(r[name],columns=['SSE_'+x for x in v.VARS]);d['origin']=o;d['model']=name;d['date']=r['days'];d['pixels_hours']=r['counts'];add('daily_sse',d)
    dest=out/'Atlas33_AUDIT.csv';tmp=dest.with_suffix('.tmp');pd.concat(frames,ignore_index=True).to_csv(tmp,index=False,float_format='%.17g');assert len(pd.read_csv(tmp,low_memory=False))==sum(map(len,frames));tmp.replace(dest)
    print('GATE',gate,flush=True);print('CSV',dest,v.sha(dest),flush=True)
if __name__=='__main__':main()
