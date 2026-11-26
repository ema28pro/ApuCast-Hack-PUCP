"""Rebuild pooled report from completed origin outputs, without fitting or reading Y."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import Atlas_v3_2_5 as a

def summarize(root):
    root=Path(root);a.check_protocol();reports=[];records={};metrics=[];weekly=[];daily=[]
    for name in a.ORIGINS:
        d=root/name;r=json.loads((d/'report.json').read_text());assert r['origin']==name and r['code_sha256']==a.v.sha(a.__file__) and r['seed_list']==a.v.SEEDS
        assert a.v.sha(d/'reference_cache.npz')==r['cache']['sha256'] and r['cache']['train_max']<r['cache']['cut']==a.ORIGINS[name][0]
        reports.append(r);records[name]=a.v.readnp(d/'daily_sse.npz');metrics.append(pd.read_csv(d/'validation_metrics.csv'));weekly.append(pd.read_csv(d/'weekly_scores.csv'));daily.append(pd.read_csv(d/'daily_scores.csv'))
    assert len({r['mode'] for r in reports})==1 and all(r['data']==reports[0]['data'] for r in reports)
    # benchmark and proxy never authorize. Neural variants are secondary; primary statistical fixed.
    mode=reports[0]['mode'];full=mode=='full';g=a.pooled_gate(records,'MixBiasVar',full)
    result=dict(mode=mode,reference=reports[0]['reference'],scope='retrospective; no blind estimate; no automatic submission',gate=g,reports=reports,preregister=a.PREREG)
    a.v.dump(root/'pooled_report.json',result)
    for name,frames in [('pooled_metrics',metrics),('pooled_weekly',weekly),('pooled_daily',daily)]:pd.concat(frames,ignore_index=True).to_csv(root/f'{name}.csv',index=False)
    # Preserve root manifest schema used by inference. Includes every completed origin.
    a.v.dump(root/'manifest.json',dict(code={Path(a.__file__).name:a.v.sha(a.__file__),'Atlas_v3_2.py':a.v.sha(a.v.__file__),'Atlas_v3_2_5_resumen.py':a.v.sha(__file__)},data=reports[0]['data'],preregister=a.PREREG,origins=a.ORIGINS,seeds=a.v.SEEDS,replay_atol=1e-4,files={str(p.relative_to(root)):a.v.sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();print(json.dumps(summarize(args.root)['gate'],indent=2))
