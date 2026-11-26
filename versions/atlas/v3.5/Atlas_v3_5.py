"""Atlas 3.5 Mutalisk Evolution. Retrospective rolling origins; manual review only."""
import os, argparse, json, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import Atlas_v3_2_5 as a
import Atlas_v3_3 as m
import Atlas_v3_4 as r
v=a.v
VARIANTS={'Long12':(48,36),'Wide64':(64,36),'Epoch54':(48,54)}
POLICIES=[(n,f) for n in VARIANTS for f in (.5,1.)]
NAMES=[f'{n}_{int(f*100)}' for n,f in POLICIES]+[n+'_ADD' for n in VARIANTS]
MIN_GAIN=.0015
WIN_SHA='4af8d9249b3ff20f8c5209f1698fd3eff71bd31dd99d5fa00bd518a2764d4870'
FINAL_ATLAS_W=np.array([.16763285710499315,.20518818054638593,.20938052051138795,.186255211148316],dtype='f4')
HIST34={'O1':.503481,'O2':.476721,'O3':.441126}
HIST27=[.45990063168146056,.4604174971675282,.4650797440837502]

def extra_windows(raw,T,old,mu,sd):
    # Exact past timestamps, never future. Missing history = current + explicit mask.
    dates=pd.to_datetime(T,format='%Y%m%d%H'); lookup={d:i for i,d in enumerate(dates)}
    indices=np.array([[lookup.get(d-pd.Timedelta(hours=k),-1) for k in (3,6,12)] for d in dates]);mask=indices>=0
    x=raw[np.where(mask,indices,np.arange(len(raw))[:,None])]
    # Store a mask frame in physical units so original normalizer yields 0/1.
    mask4=np.column_stack([mask,np.ones(len(raw),dtype=bool)]).astype('f4')
    mm=np.broadcast_to(mu+sd*mask4[:,None,None,:],raw.shape)
    return np.concatenate([old,x,mm[:,None]],axis=1).astype('f4')

def make_variant(u,st,name):
    if name=='Epoch54':return u['make']('Mutalisk',st)
    width,_=VARIANTS[name]; long=name=='Long12'; low=64 if width==48 else 80
    R=u['Residual']
    class GeoEvolution(nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('grid',torch.from_numpy(u['grid']));self.register_buffer('geo',torch.from_numpy(u['geo']));self.register_buffer('gcl',torch.from_numpy(((st['gfs_clim']-st['mu'])/st['sd']).astype('f4')))
            self.enc=nn.Sequential(R(22+(19 if long else 0),width),R(width,width,2),R(width,width));self.map=nn.Parameter(torch.zeros(1,8,70,35));self.fuse=R(width+26,width)
            self.down=nn.Sequential(R(width,low),R(low,low,2));self.up=R(width+low,width)
            self.thermal=nn.Sequential(R(width,32),nn.Conv2d(32,2,1));self.velocity=nn.Sequential(R(width,32),nn.Conv2d(32,2,1))
        def interp(self,x):
            with torch.autocast(device_type=x.device.type,enabled=False):return F.grid_sample(x.float(),self.grid.expand(len(x),-1,-1,-1),align_corners=True,padding_mode='border')
        def forward(self,x,h):
            n=len(x);fields,clock=u['inputs'](x[:,:3],h);current=x[:,2];anomaly=current-self.gcl[h.long()];av=anomaly.permute(0,3,1,2);cv=current.permute(0,3,1,2)
            inp=[fields,av,(current-x[:,0]).permute(0,3,1,2),clock]
            if long:
                mask=x[:,6,...,:3].permute(0,3,1,2).clamp(0,1); hist=x[:,3:6];cl=self.gcl[(h.long()[:,None]-torch.tensor([3,6,12],device=h.device))%24]
                ha=((hist-cl).permute(0,1,4,2,3)*mask[:,:,None]).reshape(n,12,17,13)
                delta=((current-hist[:,1])*mask[:,1,:,:,None]).permute(0,3,1,2)
                inp.extend([ha,delta,mask])
            enc=self.enc(torch.cat(inp,1));fine=self.fuse(torch.cat([self.interp(enc),self.interp(torch.cat([cv,av],1)),self.geo.expand(n,-1,-1,-1),self.map.expand(n,-1,-1,-1)],1));lo=self.down(F.avg_pool2d(fine,2));fine=self.up(torch.cat([fine,F.interpolate(lo,size=fine.shape[-2:],mode='bilinear',align_corners=False)],1))
            return torch.cat([self.thermal(fine),self.velocity(fine)],1).permute(0,2,3,1)
    return GeoEvolution()

def seed_family(u,tr,we,he,tag,name,rawq=None,Tq=None,smoke=False,query_indices=None,query_old=None):
    st=u['stats'](tr);oldW=u['W'];oldE=u['EPOCHS_NEW'];dev=u['DEVICES'];jobs=[]
    try:
        if name=='Long12':
            u['W']=extra_windows(u['X'],u['TT'],oldW,st['mu'],st['sd']);assert query_indices is not None and query_old is not None;we=extra_windows(rawq,Tq,query_old,st['mu'],st['sd'])[query_indices]
        u['EPOCHS_NEW']=36 if name=='Original' else VARIANTS[name][1]
        for seed in ([42] if smoke else [42,123]):
            torch.manual_seed(seed);model=u['make']('Mutalisk',st) if name=='Original' else make_variant(u,st,name);jobs.append((name,seed,model))
        if len(dev)==2:
            with ThreadPoolExecutor(max_workers=2) as ex:res=[f.result() for f in [ex.submit(u['fit_neural'],job,dev[i],st,tr,we,he,tag) for i,job in enumerate(jobs)]]
        else:res=[u['fit_neural'](job,dev[0],st,tr,we,he,tag) for job in jobs]
        pp=[x[1] for x in res];return np.mean(pp,axis=0,dtype='f4'),pp,[x[2] for x in res]
    finally:u['W']=oldW;u['EPOCHS_NEW']=oldE

def full_stage(u,tr,ix,tag,out,test=False):
    st=u['stats'](tr);we=(u['WT'] if test else u['W'])[ix];he=(u['HT'] if test else u['H'])[ix];orig=u['fit_neural'];seeds={}
    def capture(job,*args):
        z=orig(job,*args)
        if job[0]=='Mutalisk':seeds[job[1]]=z[1]
        return z
    u['fit_neural']=capture
    try:p=u['run_stage'](tr,we,he,tag,u['FAMILIES'])
    finally:u['fit_neural']=orig
    for name,sigma,local in [('Regional',.65,.7),('Narrow',.4,.85)]:p[name]=u['regional'](u['DEVICES'][0],st,tr,we,he,sigma=sigma,local=local,days=90)
    assert set(p)==set(u['KEYS27']);return p,seeds

def patch(base,old,new,weight,fraction):
    return v.clip(base+(new-old)*np.asarray(weight,dtype='f4')*np.float32(fraction))

def evaluate_origin(u,X,t,h,Y,T,s,o,out,rows):
    tr,ca,ev=a.split(T,h,o);ix=np.r_[ca,ev];nc=len(ca);assert [len(tr),nc,len(ev)]==m.COUNTS[o]
    assert max(T[tr])<min(T[ca])<=max(T[ca])<min(T[ev]);assert u['KEYS27'][2]=='Mutalisk'
    p,seeds=full_stage(u,tr,ix,o,out)
    def ultra(fit,query):
        params=u['calibration']({k:x[fit] for k,x in p.items()},Y[ca[fit]],'NarrowSpatial',h[ca[fit]])
        return u['apply27']({k:x[query] for k,x in p.items()},params,h[ix[query]])
    dates=v.daystr(T[ca]);oi=[];op=[]
    for day in np.unique(dates)[2:]:
        j=np.flatnonzero(dates==day);oi.extend(j);op.append(ultra(np.flatnonzero(dates<day),j))
    oi=np.asarray(oi);boof=np.concatenate(op);params=u['calibration']({k:x[:nc] for k,x in p.items()},Y[ca],'NarrowSpatial',h[ca]);bp=ultra(np.arange(nc),np.arange(len(ix)))
    phys=a.fit_anchor(X,t,h,Y,T,tr,s);ph=a.predict_anchor(phys,X[ix],t[ix],h[ix],T[ix]);old,pre=a.corrected(ph,ca,ix,h,Y,T,oi);w,info=a.choose_weights(boof,pre,Y[ca[oi]],v.daystr(T[ca[oi]]),True)
    rp,_=r.components(X,t,h,Y,T,s,tr,ca,ev);base=v.blend(bp[nc:],rp['RegimeFlow'],w)
    if abs(v.score(base,Y[ev])-HIST34[o])>.001:raise RuntimeError(f'{o} incumbent outside historical tolerance: {v.score(base,Y[ev])}')
    # Both control and intervention retain same calibration and Atlas branch.
    eff=(1-w)*params[0][u['KEYS27'].index('Mutalisk')];pred={'Reference':base};oldmut=p['Mutalisk'];variants={}
    for n in VARIANTS:
        q,ss,rr=seed_family(u,tr,u['W'][ix],h[ix],o,n,u['X'],T,query_indices=ix,query_old=u['W'])
        variants[n]=q
        # Additional convex component, weights fit only on prequential calibration days.
        addw,addinfo=a.choose_weights(boof,q[oi],Y[ca[oi]],v.daystr(T[ca[oi]]),True)
        pred[n+'_ADD']=v.blend(v.blend(bp[nc:],q[nc:],addw),rp['RegimeFlow'],w)
        rows.append(dict(section='add_weights',origin=o,model=n+'_ADD',value=json.dumps({'weights':addw.tolist(),**addinfo})))
        for k,f in POLICIES:
            if k==n:pred[f'{n}_{int(f*100)}']=v.blend(patch(bp[nc:],oldmut[nc:],q[nc:],params[0][2],f),rp['RegimeFlow'],w)
        for i,x in enumerate(ss):rows.append(dict(section='seed',origin=o,model=n,seed=rr[i]['seed'],skill=v.score(x[nc:],Y[ev]),calibration_skill=v.score(x[:nc],Y[ca]),seconds=rr[i]['seconds']))
        for j,k in enumerate(v.VARS):
            e=(ss[0][nc:,...,j]-Y[ev,...,j]).ravel();f=(ss[1][nc:,...,j]-Y[ev,...,j]).ravel();rows.append(dict(section='seed_correlation',origin=o,model=n,variable=k,pearson=float(np.corrcoef(e,f)[0,1])))
    for seed,q in sorted(seeds.items()):rows.append(dict(section='seed',origin=o,model='Original',seed=seed,skill=v.score(q[nc:],Y[ev]),calibration_skill=v.score(q[:nc],Y[ca])))
    for j,k in enumerate(v.VARS):
        e=(seeds[42][nc:,...,j]-Y[ev,...,j]).ravel();f=(seeds[123][nc:,...,j]-Y[ev,...,j]).ravel();rows.append(dict(section='seed_correlation',origin=o,model='Original',variable=k,pearson=float(np.corrcoef(e,f)[0,1])))
    for n,q in [('Original',oldmut),*variants.items()]:rows.append(dict(section='family',origin=o,model=n,skill=v.score(q[nc:],Y[ev]),calibration_skill=v.score(q[:nc],Y[ca])))
    dst=out/o;dst.mkdir(exist_ok=True);a.metrics(pred,Y[ev],T[ev],o,dst);rec=v.readnp(dst/'daily_sse.npz');rows.extend(dict(section='global',**x) for x in pd.read_csv(dst/'validation_metrics.csv').to_dict('records'))
    for n in pred:
        for d,c,ss in zip(rec['days'],rec['counts'],rec[n]):rows.append(dict(section='daily_sse',origin=o,model=n,date=d,pixels_hours=int(c),**dict(zip(['SSE_'+x for x in v.VARS],ss))))
    rows.append(dict(section='weights',origin=o,value=json.dumps({'atlas':w.tolist(),'ultra_mutalisk':params[0][2].tolist(),'effective_mutalisk':eff.tolist()})))
    print(o, pd.read_csv(dst/'validation_metrics.csv')[['model','skill']].to_string(index=False),flush=True);return rec

def gain(records,name):
    ss=sum(q[name].sum(0) for q in records);b=sum(q['Reference'].sum(0) for q in records);n=sum(q['counts'].sum() for q in records);return float(v.skill(ss,n)-v.skill(b,n))

def select(records):
    ok=[n for n in NAMES if min(gain([r],n) for r in records.values())>0];return max(ok,key=lambda n:gain(list(records.values()),n)) if ok else None

def check_export(data,values,path):
    tpl=pd.read_csv(data/'sample_submission.csv',dtype={'id':str});assert values.shape==(len(tpl),4) and np.isfinite(values).all();values=values.copy();values[:,0]=np.clip(values[:,0],0,100)
    d=pd.DataFrame(values,columns=v.VARS);d.insert(0,'id',tpl.id);tmp=path.with_suffix('.tmp');d.to_csv(tmp,index=False,float_format='%.6f');z=pd.read_csv(tmp,dtype={'id':str});assert z.id.equals(tpl.id) and z.id.is_unique and z.shape==(646800,5) and np.isfinite(z[v.VARS]).all().all() and z.RH.between(0,100).all();tmp.replace(path);return v.sha(path)

def deploy(u,data,X,t,h,Y,T,s,selected,out,rows):
    name=selected.rsplit('_',1)[0];suffix=selected.rsplit('_',1)[1];f=0. if suffix=='ADD' else int(suffix)/100;tr=np.argsort(T);ev=np.unique(u['ti']);q=u['WT'][ev];he=u['HT'][ev]
    # Rebuild original v2.7 final recipe with original 3 calibration weeks.
    cp=[];cy=[];ch=[]
    for i,(cut,cal,end) in enumerate(u['CUTS']):
        it=np.flatnonzero(T<cut);ic=np.flatnonzero((T>=cut)&(T<cal)&(h%3==0));ie=np.flatnonzero((T>=cal)&(T<end)&(h%3==0));p,_=full_stage(u,it,np.r_[ic,ie],f'D{i+1}',out);nc=len(ic)
        pc={k:x[:nc] for k,x in p.items()};params=u['calibration'](pc,Y[ic],'NarrowSpatial',h[ic]);skill=v.score(u['apply27']({k:x[nc:] for k,x in p.items()},params,h[ie]),Y[ie]);assert abs(skill-HIST27[i])<=.001,(i,skill);cp.append(pc);cy.append(Y[ic]);ch.append(h[ic])
    params=u['calibration']({k:np.concatenate([p[k] for p in cp]) for k in u['KEYS27']},np.concatenate(cy),'NarrowSpatial',np.concatenate(ch))
    p,_=full_stage(u,tr,ev,'final',out,True);base=u['apply27'](p,params,he);new,_,_=seed_family(u,tr,q,he,'deploy',name,u['XT'],u['TE'],query_indices=ev,query_old=u['WT']);addw=None
    if suffix=='ADD':
        addw=np.asarray(json.loads(next(row['value'] for row in rows if row['section']=='add_weights' and row['origin']=='O3' and row['model']==selected))['weights'],dtype='f4');candidate=v.blend(base,new,addw)
    else:candidate=patch(base,p['Mutalisk'],new,params[0][2],f)
    te=v.readnp(data/'test.npz');assert 'Y' not in te;XX,tt,hh=v.A['prepare'](te['X'],te['times'].astype(str),s);it=np.flatnonzero(T<'2024062400');ic=np.flatnonzero((T>='2024062400')&(h%3==0));assert len(it)==2767 and len(ic)==56
    ap,_=r.components(X,t,h,Y,T,s,it,ic,ev,XX,tt,hh,te['times'].astype(str));positions=np.full(len(u['TE']),-1,int);positions[ev]=np.arange(len(ev));order=(positions[u['ti']],u['ri'],u['ci']);base34=v.blend(base.round(4),ap['RegimeFlow'],FINAL_ATLAS_W);new34=v.blend(candidate.round(4),ap['RegimeFlow'],FINAL_ATLAS_W);values=new34[order]
    # If exact winning file is attached, audit the reconstructed control against it, not use labels.
    matches=[]
    for path in Path('/kaggle/input').rglob('*.csv'):
        if path.name.startswith('Atlas34_REVIEW') and v.sha(path)==WIN_SHA:matches.append(path)
    control_mae=None
    if matches:
        win=pd.read_csv(matches[0],dtype={'id':str});assert win.id.equals(u['TEMPLATE'].id);control_mae=np.abs(win[v.VARS].to_numpy()-base34[order]).mean(0).tolist()
    digest=check_export(data,values,out/'Atlas35_REVIEW.csv');return dict(csv_sha256=digest,rows=len(values),selected=selected,base_mode='matched_reconstruction',control_mae_vs_public=control_mae,weights=params[0][2].tolist(),atlas_weights=FINAL_ATLAS_W.tolist(),add_weights=None if addw is None else addw.tolist(),mean_abs_change=np.abs(values-base34[order]).mean(0).tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'Atlas35_REVIEW.csv').unlink(missing_ok=True)
    data=v.data_root(args.data);assert v.fingerprint(data)==m.OFFICIAL_HASHES
    for name,digest in m.EXPECTED.items():assert v.sha(Path(__file__).with_name(name))==digest
    a.check_protocol();m.tests();train=v.readnp(data/'train.npz');s=v.readnp(data/'static.npz');T=train['times'].astype(str);Y=train['Y'];X,t,h=v.A['prepare'](train['X'],T,s);u=v.init_ultra(data,out,'smoke' if args.smoke else 'full')
    if args.smoke:
        tr=np.arange(32);ix=np.arange(80,84);st=u['stats'](tr);z=extra_windows(u['X'],T,u['W'],st['mu'],st['sd']);assert z.shape==(len(T),7,17,13,4);assert np.allclose(z[0,6,...,:3],st['mu'][:3]);assert np.array_equal(z[:,:3],u['W']);rows=[]
        for n in ['Original',*VARIANTS]:
            q,_,rr=seed_family(u,tr,u['W'][ix],h[ix],'smoke',n,u['X'],T,smoke=True,query_indices=ix,query_old=u['W']);assert q.shape==(4,70,35,4) and np.isfinite(q).all();assert np.array_equal(patch(q,q,q,[.2]*4,1),q);rows.append(dict(model=n,skill_smoke_not_evidence=v.score(q,Y[ix]),seconds=rr[0]['seconds']))
        rng=np.random.default_rng(35); yy=Y[ix];pp={k:v.clip(yy+rng.normal(0,.5,yy.shape).astype('f4')) for k in u['KEYS27']};params=u['calibration'](pp,yy,'NarrowSpatial',h[ix]);bp=u['apply27'](pp,params,h[ix]);assert bp.shape==yy.shape and np.isfinite(bp).all();assert np.allclose(params[0].sum(0),1)
        assert np.array_equal(v.blend(bp,bp,[.25]*4),bp);assert np.array_equal(patch(bp,pp['Mutalisk'],pp['Mutalisk'],params[0][2],1),bp)
        pd.DataFrame(rows).to_csv(out/'smoke.csv',index=False);print('SMOKE OK: no submission, no performance inference',flush=True);return
    torch.use_deterministic_algorithms(True,warn_only=True);records={};rows=[];selected=None
    def save():pd.DataFrame(rows).to_csv(out/'Atlas35_FULL_AUDIT.csv',index=False,float_format='%.17g')
    for o in a.ORIGINS:
        if o=='O3':
            selected=select(records);rows.append(dict(section='selection_before_O3',value=json.dumps({'selected':selected,'dev_gains':{n:{k:gain([q],n) for k,q in records.items()} for n in NAMES}})));save();print('LOCKED BEFORE O3',selected,flush=True)
        records[o]=evaluate_origin(u,X,t,h,Y,T,s,o,out,rows);save()
    for n in NAMES:rows.append(dict(section='pooled',model=n,gain=gain(list(records.values()),n)))
    gate=None;passed=False
    if selected:
        gate=a.pooled_gate(records,selected,True);passed=bool(gate['review_eligible'] and min(gate['origins'].values())>0 and gate['pooled_gain']>=MIN_GAIN and gate['last_gain']>=MIN_GAIN);gate['retrospective_deploy_gate']=passed;gate['review_eligible']=False
    rows.append(dict(section='gate',value=json.dumps({'selected':selected,'gate':gate,'min_gain':MIN_GAIN,'scope':'retrospective only'})));save();print('GATE',gate,flush=True)
    if passed:
        report=deploy(u,data,X,t,h,Y,T,s,selected,out,rows);rows.append(dict(section='deploy',value=json.dumps(report)));save();print('EXPERIMENTAL CSV',report,flush=True)
    else:print('NO SUBMIT: keep Atlas 3.4',flush=True)
if __name__=='__main__':main()
