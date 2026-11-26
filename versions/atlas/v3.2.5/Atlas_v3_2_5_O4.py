"""Review only: O4 fits Atlas Bias on official data, with O3 weights frozen."""
import argparse, json
from pathlib import Path
import Atlas_v3_2_5 as a
import numpy as np
import pandas as pd
v=a.v
VARS=['RH','T','U','V']
BASE_SHA='9ce6e7594d0edbf884e56902fd67e1fa02de1a94ba2f6815fc5c8b5115a801e1'
AUDIT_SHA='724ca9c3b31bd82852e2019e3db4dc3d2439d0188092c9868f7a1ba8ecfb17c7'
CUT='2024062400'; END='2024070100'
def main():
 p=argparse.ArgumentParser()
 for k in ['data','audit','base','out']: p.add_argument('--'+k,required=True)
 args=p.parse_args(); data=Path(args.data); out=Path(args.out);out.mkdir(parents=True,exist_ok=True);base=Path(args.base);audit=Path(args.audit)
 assert v.sha(base)==BASE_SHA and v.sha(audit)==AUDIT_SHA
 a.check_protocol();d=pd.read_csv(audit,low_memory=False)
 def meta(section,origin=None):
  q=d[d.section==section];q=q if origin is None else q[q.origin==origin];return {r.field:json.loads(r.value) for r in q.itertuples() if pd.notna(r.value)}
 m=meta('manifest');dh=v.fingerprint(data);assert dh==m['data'];assert m['code']=={'Atlas_v3_2.py':v.sha(v.__file__),'Atlas_v3_2_5.py':v.sha(a.__file__)}
 records={}
 for o in a.ORIGINS:
  r=meta('report',o);c=meta('cache_metadata',o);assert r['mode']=='full' and r['reference']=='reconstructed v2.7' and c['mode']=='full' and c['train_max']<c['cut'] and c['data']==dh and r['code_sha256']==v.sha(a.__file__)
  q=d[(d.section=='daily_sse')&(d.origin==o)];x=q[q.model=='Reference'].sort_values('day');y=q[q.model=='MixBiasVar'].sort_values('day');assert np.array_equal(x.date,y.date) and np.array_equal(x.pixels_hours,y.pixels_hours)
  columns=['SSE_'+j for j in VARS];records[o]={'counts':x.pixels_hours.to_numpy(dtype='int64'),'Reference':x[columns].to_numpy(),'MixBiasVar':y[columns].to_numpy()}
 gate=a.pooled_gate(records,'MixBiasVar',True);assert gate['review_eligible'];w=np.array(meta('report','O3')['policies']['MixBiasVar']['weights']);assert w.shape==(4,) and np.all((w>=0)&(w<=.5));print('Gate:',gate,flush=True);print('Frozen weights:',w,flush=True)
 train=v.readnp(data/'train.npz');s=v.readnp(data/'static.npz');T=train['times'].astype(str);Y=train['Y'];X,t,h=v.A['prepare'](train['X'],T,s);tr=np.flatnonzero(T<CUT);ca=np.flatnonzero((T>=CUT)&(T<END)&(h%3==0));assert len(tr) and len(ca) and max(T[tr])<min(T[ca]);assert not np.intersect1d(tr,ca).size
 st=a.fit_anchor(X,t,h,Y,T,tr,s,'physics',False,False);pc=a.predict_anchor(st,X[ca],t[ca],h[ca],T[ca]);bias=v.bias(pc,Y[ca],h[ca]);np.savez(out/'anchor_Physics.npz',**st);np.savez(out/'anchor_bias.npz',bias=bias);np.savez(out/'static.npz',**s)
 test=v.readnp(data/'test.npz');TT=test['times'].astype(str);XX,tt,hh=v.A['prepare'](test['X'],TT,s);assert min(TT)>=END and len(TT)==792;ix=np.flatnonzero(hh%3==0);assert len(ix)==264
 pred=v.clip(a.predict_anchor(st,XX[ix],tt[ix],hh[ix],TT[ix])+bias[hh[ix]])
 replay=v.clip(a.predict_anchor(v.readnp(out/'anchor_Physics.npz'),XX[ix[:8]],tt[ix[:8]],hh[ix[:8]],TT[ix[:8]])+v.readnp(out/'anchor_bias.npz')['bias'][hh[ix[:8]]]);assert np.max(np.abs(replay-pred[:8]))<=1e-4
 template=pd.read_csv(data/'sample_submission.csv',dtype={'id':str});b=pd.read_csv(base,dtype={'id':str});ids=np.array([f'{stamp}_{i:02d}_{j:02d}' for stamp in TT[ix] for i in range(70) for j in range(35)]);assert len(ids)==646800 and np.array_equal(ids,template.id.to_numpy()) and np.array_equal(ids,b.id.to_numpy()) and b.columns.tolist()==['id']+VARS
 bval=b[VARS].to_numpy();aval=pred.reshape(-1,4).astype('float64');assert np.isfinite(bval).all();mixed=bval+(aval-bval)*w;mixed[:,0]=np.clip(mixed[:,0],0,100);assert np.isfinite(mixed).all();result=pd.DataFrame(mixed,columns=VARS);result.insert(0,'id',ids);dest=out/'REVIEW.csv';result.to_csv(dest,index=False,float_format='%.6f');check=pd.read_csv(dest,dtype={'id':str});assert len(check)==646800 and np.array_equal(check.id,template.id) and np.isfinite(check[VARS]).all().all() and check.RH.between(0,100).all()
 mad=np.abs(check[VARS].to_numpy()-bval).mean(0);rawmad=np.abs(aval-bval).mean(0);manifest=dict(origin='O4',scope='review only; no test labels; retrospective validation',cut=CUT,train_count=len(tr),train_max=str(max(T[tr])),cal_count=len(ca),cal_times=T[ca].tolist(),test_count=len(ix),test_min=str(min(TT[ix])),test_max=str(max(TT[ix])),weights=dict(zip(VARS,w.tolist())),mean_abs_change=dict(zip(VARS,mad.tolist())),mean_abs_component_gap=dict(zip(VARS,rawmad.tolist())),cal_uncorrected_skill=v.score(pc,Y[ca]),data=dh,base_sha256=BASE_SHA,audit_sha256=AUDIT_SHA,code={**m['code'],Path(__file__).name:v.sha(__file__)},gate=gate,files={f:v.sha(out/f) for f in ['anchor_Physics.npz','anchor_bias.npz','static.npz','REVIEW.csv']})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
