"""Atlas v3.4 RegimeFlow: retrospective variable routing; no automatic submission."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import Atlas_v3_3 as m
import Atlas_v3_2_5 as a
v=a.v; VARS=v.VARS; SSE=['SSE_'+x for x in VARS]
AUDIT_SHA='78568dd2c40e4e5d4317fc113fe992eff4e8f4143b9d8c0bbec6d6cbcd9d83a7'
BASE_SHA='9ce6e7594d0edbf884e56902fd67e1fa02de1a94ba2f6815fc5c8b5115a801e1'
CODE_SHA='58015cb53e74b0f55c4fc1136a98851e7e91f184ea78634359ddca420df09377'
ROUTE=('MixDriftBias','MixDriftBias','MixAdaptiveFlow','MixDriftBias') # post-hoc, NOT confirmatory
NAMES=['Ultralisk','Reference','MixAdaptiveFlow','MixDriftBias','MixRecentBias','RegimeFlow']
def audit(path):
    assert v.sha(path)==AUDIT_SHA
    d=pd.read_csv(path,low_memory=False);meta={r.field:json.loads(r.value) for r in d[d.section=='metadata'].itertuples()}
    assert meta['mode']=='full' and meta['source_sha256']==CODE_SHA and meta['data']==m.OFFICIAL_HASHES and meta['historical_audit_sha256']==m.FULL_AUDIT_SHA
    assert {r['origin'] for r in meta['reports']}==set(a.ORIGINS)
    records={};rows=[]
    for r in meta['reports']:
        o=r['origin'];assert r['counts']==m.COUNTS[o] and r['mode']=='full' and r['data']==meta['data'] and r['code_sha256']==CODE_SHA and r['reference']=='MixBiasVar / reconstructed Ultralisk v2.7'
        assert r['cache']['train_max']<r['cache']['cut']==a.ORIGINS[o][0]
        w=np.asarray(r['weights']);assert w.shape==(4,) and np.all((w>=0)&(w<=.5))
        q=d[(d.section=='daily_sse')&(d.origin==o)];rec={}
        days=pd.date_range(pd.to_datetime(a.ORIGINS[o][1],format='%Y%m%d%H'),pd.to_datetime(a.ORIGINS[o][2],format='%Y%m%d%H')-pd.Timedelta(days=1)).strftime('%Y%m%d').astype(int).to_numpy()
        for n in NAMES[:-1]:
            x=q[q.model==n].sort_values('date');assert np.array_equal(x.date.to_numpy(),days)
            ns=x.pixels_hours.to_numpy(dtype='int64');ss=x[SSE].to_numpy(dtype='f8');assert np.isfinite(ss).all() and (ss>=0).all() and (ns>=0).all() and (ss[ns==0]==0).all()
            if 'counts' in rec:assert np.array_equal(rec['counts'],ns)
            rec['counts']=ns;rec[n]=ss
            g=d[(d.section=='global')&(d.origin==o)&(d.model==n)];assert len(g)==1 and abs(float(v.skill(ss.sum(0),ns.sum()))-g.skill.iloc[0])<1e-9
        assert rec['counts'].sum()==m.COUNTS[o][2]*2450
        rec['RegimeFlow']=np.stack([rec[n][:,j] for j,n in enumerate(ROUTE)],axis=1);records[o]=rec
    def row(o,n,ss,counts,section='global',**extra):
        rm=np.sqrt(ss/counts);return dict(section=section,origin=o,model=n,skill=float(v.skill(ss,counts)),pixels_hours=int(counts),**dict(zip(['NRMSE_'+j for j in VARS],rm/v.NORM)),**extra)
    for o,r in records.items():
        for n in NAMES:
            rows.append(row(o,n,r[n].sum(0),r['counts'].sum()))
            for i in range(0,len(r['counts']),7):
                ns=r['counts'][i:i+7].sum()
                if ns:rows.append(row(o,n,r[n][i:i+7].sum(0),ns,'weekly',week=i//7+1))
            for i,ns in enumerate(r['counts']):
                if ns:rows.append(row(o,n,r[n][i],ns,'daily',day=i+1))
                rows.append(dict(section='daily_sse',origin=o,model=n,day=i+1,pixels_hours=int(ns),**dict(zip(SSE,r[n][i]))))
    for n in NAMES:rows.append(row('Pooled',n,sum(r[n].sum(0) for r in records.values()),sum(r['counts'].sum() for r in records.values())))
    gate=a.pooled_gate(records,'RegimeFlow',True);gate['statistical_gate_passed_retrospective']=bool(gate['review_eligible'] and min(gate['origins'].values())>0);gate['review_eligible']=False
    return meta,rows,gate

def components(X,t,h,Y,T,s,tr,ca,query,XX=None,tt=None,hh=None,TT=None):
    assert max(T[tr])<min(T[ca]); assert not np.intersect1d(tr,ca).size
    if XX is None: XX,tt,hh,TT=X,t,h,T
    assert max(T[ca])<min(TT[query]);pred={};states={}
    for name,drift in [('Physics',False),('Drift',True)]:
        st=a.fit_anchor(X,t,h,Y,T,tr,s,'physics',False,drift)
        p=a.predict_anchor(st,X[ca],t[ca],h[ca],T[ca]);q=a.predict_anchor(st,XX[query],tt[query],hh[query],TT[query]);states[name]=st
        variant='AdaptiveFlow' if not drift else 'Bias';corr=m.fit_correction(p,Y[ca],h[ca],variant);last=pd.to_datetime(T[ca[-1]],format='%Y%m%d%H')
        pred['AdaptiveFlow' if not drift else 'DriftBias']=m.apply_correction(corr,q,hh[query],TT[query],last,variant);states[variant]=corr
        if not drift:pred['Bias325']=v.clip(q+v.bias(p,Y[ca],h[ca])[hh[query]])
    pred['RegimeFlow']=pred['DriftBias'].copy();pred['RegimeFlow'][...,2]=pred['AdaptiveFlow'][...,2]
    return pred,states

def diagnostics(X,t,h,Y,T,s,train,test,TT):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.decomposition import PCA
    from scipy.stats import ks_2samp
    def feat(x):
        q=np.concatenate([x,np.hypot(x[...,2],x[...,3])[...,None]],axis=-1)
        return np.concatenate([q.mean((1,2)),q.std((1,2)),np.quantile(q,.1,axis=(1,2)),np.quantile(q,.9,axis=(1,2))],axis=1)
    t0=pd.to_datetime(T,format='%Y%m%d%H');t1=pd.to_datetime(TT,format='%Y%m%d%H');it=(t0>=t0.max().normalize()-pd.Timedelta(days=33))&(t0.hour%3==0);ie=t1.hour%3==0
    f0,f1=feat(train[it]),feat(test[ie]);F=np.r_[f0,f1];y=np.r_[np.zeros(len(f0),int),np.ones(len(f1),int)]
    groups=np.r_[np.asarray((t0[it]-t0[it].min()).days)//7,100+np.asarray((t1[ie]-t1[ie].min()).days)//7]
    blocks0=np.unique(groups[y==0]);blocks1=np.unique(groups[y==1]);assert len(blocks0)==len(blocks1)
    prob=np.full(len(y),np.nan);auc=[]
    for b0,b1 in zip(blocks0,blocks1):
        ev=np.isin(groups,[b0,b1]);tr=~ev;clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000));clf.fit(F[tr],y[tr]);prob[ev]=clf.predict_proba(F[ev])[:,1];auc.append(float(roc_auc_score(y[ev],prob[ev])))
    rows=[dict(section='shift',metric=k,value=x) for k,x in [('adversarial_auc_oof',float(roc_auc_score(y,prob))),('auc_fold_min',min(auc)),('auc_fold_max',max(auc)),('n_train',len(f0)),('n_test',len(f1)),('blocks',len(np.unique(groups)))]]
    for j,n in enumerate([*VARS,'speed']):
        for hour in range(0,24,3):rows.append(dict(section='shift',metric='KS_spatial_mean',variable=n,hour=hour,value=float(ks_2samp(f0[t0[it].hour==hour,j],f1[t1[ie].hour==hour,j]).statistic)))
    x=train[it].reshape(len(f0),-1);z=test[ie].reshape(len(f1),-1);sc=StandardScaler().fit(x);xp=sc.transform(x);zp=sc.transform(z);pca=PCA(n_components=12,random_state=34,svd_solver='randomized').fit(xp)
    for n,arr in [('train',xp),('test',zp)]:rows.append(dict(section='shift',metric='EOF_reconstruction_MSE',period=n,value=float(np.mean((arr-pca.inverse_transform(pca.transform(arr)))**2))))
    dem=s['dem'];sl=np.hypot(*np.gradient(dem))
    for o in a.ORIGINS:
        tr,ca,ev=a.split(T,h,o);pred,_=components(X,t,h,Y,T,s,tr,ca,ev);clim=v.clip(v.A['climate'](Y[tr],h[tr],np.arange(len(tr)))[h[ev]])
        for n,q in [('Climatology',clim),*pred.items()]:rows.append(dict(section='anchor_only',origin=o,model=n,skill=v.score(q,Y[ev])))
        p=pred['Bias325'];y=Y[ev];speed=np.hypot(p[...,2],p[...,3]);ts=np.hypot(y[...,2],y[...,3]);ang=np.arctan2(p[...,3],p[...,2])-np.arctan2(y[...,3],y[...,2]);ang=np.abs(np.arctan2(np.sin(ang),np.cos(ang)))*180/np.pi
        masks={**{f'hour_{i:02}':np.broadcast_to((h[ev]==i)[:,None,None],speed.shape) for i in range(0,24,3)},'wind_lt2':speed<2,'wind_2_5':(speed>=2)&(speed<5),'wind_ge5':speed>=5}
        for name,field in [('slope',sl),('altitude',dem)]:
            l,u=np.quantile(field,[.33,.67]);
            for i,mask in enumerate([field<=l,(field>l)&(field<=u),field>u]):masks[f'{name}_{i+1}']=np.broadcast_to(mask,speed.shape)
        for label,mask in masks.items():
            if not mask.any():continue
            keep=mask&(ts>=1)&(speed>=1);rows.append(dict(section='wind_anchor_only',origin=o,regime=label,pixels_hours=int(mask.sum()),RMSE_U=float(np.sqrt(np.mean((p[...,2][mask]-y[...,2][mask])**2))),RMSE_V=float(np.sqrt(np.mean((p[...,3][mask]-y[...,3][mask])**2))),speed_RMSE=float(np.sqrt(np.mean((speed[mask]-ts[mask])**2))),direction_MAE_deg=float(ang[keep].mean()) if keep.any() else np.nan,direction_count=int(keep.sum())))
        print('diagnostics',o,flush=True)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--audit',required=True);p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--diagnostics',action='store_true');args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);data=v.data_root(args.data);dh=v.fingerprint(data);assert dh==m.OFFICIAL_HASHES
    a.check_protocol();m.tests();assert v.sha(m.__file__)==CODE_SHA
    for name,digest in m.EXPECTED.items():assert v.sha(Path(__file__).with_name(name))==digest
    meta,rows,gate=audit(Path(args.audit));print('RETROSPECTIVE GATE:',json.dumps(gate),flush=True)
    assert gate['statistical_gate_passed_retrospective'],'No competitive file: retrospective gate failed'
    train=v.readnp(data/'train.npz');s=v.readnp(data/'static.npz');test=v.readnp(data/'test.npz');assert 'Y' not in test;T=train['times'].astype(str);TT=test['times'].astype(str);Y=train['Y'];X,t,h=v.A['prepare'](train['X'],T,s)
    if args.diagnostics:rows+=diagnostics(X,t,h,Y,T,s,train['X'],test['X'],TT)
    base=Path(args.base);assert v.sha(base)==BASE_SHA
    tr=np.flatnonzero(T<'2024062400');ca=np.flatnonzero((T>='2024062400')&(T<'2024070100')&(h%3==0));assert len(tr)==2767 and len(ca)==56 and max(T[tr])<min(T[ca]);assert len(TT)==792
    XX,tt,hh=v.A['prepare'](test['X'],TT,s);ix=np.flatnonzero(hh%3==0);assert len(ix)==264 and min(TT)>='2024070100'
    pred,states=components(X,t,h,Y,T,s,tr,ca,ix,XX,tt,hh,TT)
    w=np.asarray(next(r['weights'] for r in meta['reports'] if r['origin']=='O3'),dtype='f8')
    tpl=pd.read_csv(data/'sample_submission.csv',dtype={'id':str});b=pd.read_csv(base,dtype={'id':str});ids=np.array([f'{stamp}_{i:02d}_{j:02d}' for stamp in TT[ix] for i in range(70) for j in range(35)]);assert len(ids)==646800 and np.array_equal(ids,tpl.id.to_numpy()) and np.array_equal(ids,b.id.to_numpy()) and list(b.columns)==['id',*VARS]
    bv=b[VARS].to_numpy();assert np.isfinite(bv).all() and b.RH.between(0,100).all();av=pred['RegimeFlow'].reshape(-1,4);inc=pred['Bias325'].reshape(-1,4)
    mix=bv+(av-bv)*w;mix[:,0]=np.clip(mix[:,0],0,100);prev=bv+(inc-bv)*w;prev[:,0]=np.clip(prev[:,0],0,100)
    dd=bv+(pred['DriftBias'].reshape(-1,4)-bv)*w;aa=bv+(pred['AdaptiveFlow'].reshape(-1,4)-bv)*w;dd[:,2]=aa[:,2];dd[:,0]=np.clip(dd[:,0],0,100);assert np.array_equal(dd,mix)
    for name,st in states.items():np.savez(out/f'{name}.npz',**st)
    last=pd.to_datetime(T[ca[-1]],format='%Y%m%d%H');ii=ix[:8];p=a.predict_anchor(v.readnp(out/'Physics.npz'),XX[ii],tt[ii],hh[ii],TT[ii]);q=a.predict_anchor(v.readnp(out/'Drift.npz'),XX[ii],tt[ii],hh[ii],TT[ii]);u=m.apply_correction(v.readnp(out/'AdaptiveFlow.npz'),p,hh[ii],TT[ii],last,'AdaptiveFlow');q=m.apply_correction(v.readnp(out/'Bias.npz'),q,hh[ii],TT[ii],last,'Bias');q[...,2]=u[...,2];err=float(np.max(np.abs(q-pred['RegimeFlow'][:8])));assert err<=1e-4
    result=pd.DataFrame(mix,columns=VARS);result.insert(0,'id',ids);dest=out/'Atlas34_REVIEW.csv';tmp=dest.with_suffix('.tmp');result.to_csv(tmp,index=False,float_format='%.6f');check=pd.read_csv(tmp,dtype={'id':str});assert len(check)==646800 and np.array_equal(check.id,ids) and list(check.columns)==['id',*VARS] and np.isfinite(check[VARS]).all().all() and check.RH.between(0,100).all();tmp.replace(dest)
    report=dict(train_count=len(tr),cal_count=len(ca),train_max=str(max(T[tr])),cal_min=str(min(T[ca])),cal_max=str(max(T[ca])),test_count=len(ix),weights=dict(zip(VARS,w.tolist())),mean_abs_vs_v27=dict(zip(VARS,np.abs(check[VARS].to_numpy()-bv).mean(0).tolist())),mean_abs_vs_same_weights_bias325=dict(zip(VARS,np.abs(check[VARS].to_numpy()-prev).mean(0).tolist())),replay_max_abs=err,csv_sha256=v.sha(dest),rows=len(check),scope='manual experimental review; no test labels; no automatic submission')
    rows+=[dict(section='metadata',field=k,value=json.dumps(val,sort_keys=True)) for k,val in dict(route=dict(zip(VARS,ROUTE)),gate=gate,deployment=report,source_sha256=v.sha(__file__),audit_sha256=AUDIT_SHA,data=dh).items()]
    pd.DataFrame(rows).to_csv(out/'Atlas34_AUDIT.csv',index=False,float_format='%.17g');manifest=dict(version='3.4',route=dict(zip(VARS,ROUTE)),gate=gate,deploy=report,source_sha256=v.sha(__file__),audit_sha256=AUDIT_SHA,base_sha256=BASE_SHA,data=dh,files={p.name:v.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'})
    v.dump(out/'manifest.json',manifest);print('DEPLOY:',json.dumps(report),flush=True)
if __name__=='__main__':main()
