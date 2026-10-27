"""Small seed-averaging study. Preserve V4; select on val_00..02 only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import uwb_detector as b
import uwb_search_v4 as v4

HERE=Path(__file__).resolve().parent
OUT=HERE/'search_v5'

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--predict',action='store_true');p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args()
    OUT.mkdir(exist_ok=True);sys.path.insert(0,str(a.data/'src'));import abc2027.evaluate as ev
    if a.predict:
        from threadpoolctl import threadpool_limits
        v4.ROOT=OUT;a.cache=HERE/'run/cache'
        with threadpool_limits(limits=4):v4.predict(a,ev)
        return
    sources={
        'motion':(HERE/'search_v4/motion_only/val_probabilities.npz',HERE/'search_v4/motion_only/model.joblib','tree','motion'),
        'seed28':(HERE/'search_v4/tcn16_seed2028/val_probabilities.npz',HERE/'search_v4/tcn16_seed2028/model.pt','tcn','base'),
        'seed29':(HERE/'search_v4/tcn16_seed2029/val_probabilities.npz',HERE/'search_v4/tcn16_seed2029/model.pt','tcn','base'),
        'seed27':(HERE/'temporal_v2/val_probabilities.npz',HERE/'temporal_v2/model.pt','tcn','base')}
    probs={n:v4.read_probs(s[0]) for n,s in sources.items()}
    dur=ev.stream_durations(a.data/'streams/val');td={s:d for s,d in dur.items() if s in ['val_00','val_01','val_02']};dd={s:d for s,d in dur.items() if s in ['val_03','val_04']}
    gt=ev.load_ground_truth(a.data/'streams_events.csv',True)
    baseline=json.loads((HERE/'search_v4/selected.json').read_text())
    baseline_p=v4.read_probs(HERE/'search_v4/selected_probabilities.npz')
    baseline_rows=[r for s in td for r in v4.decode_rows(baseline_p[s],s,dur[s],baseline)]
    best=(b.ranking_key(ev.evaluate(gt,baseline_rows,td)),dict(baseline,name='retained_v4'),baseline_p)
    trials=[]
    for seeds in [('seed29',),('seed28','seed29'),('seed27','seed28','seed29')]:
        for tree_weight in (.35,.5,.65):
            name='_'.join(seeds)+'_motion'+str(tree_weight)
            weights={'motion':tree_weight,**{s:(1-tree_weight)/len(seeds) for s in seeds}}
            mix={sid:sum(w*probs[n][sid] for n,w in weights.items()) for sid in dur}
            cfg,decoder_trials=b.tune({s:mix[s] for s in td},td,gt,ev.evaluate)
            components=[dict(name=n,path=str(sources[n][1]),kind=sources[n][2],features=sources[n][3],weight=w) for n,w in weights.items()]
            config=dict(name=name,decoder=cfg,boundary=None,alarm_delay=0.,components=components)
            rows=[r for s in td for r in v4.decode_rows(mix[s],s,dur[s],config)]
            report=ev.evaluate(gt,rows,td);key=b.ranking_key(report)
            trials.append(dict(config=config,ranking=report['ranking_score'],decoder_trials=decoder_trials))
            print(name,'tuning F1',round(key[0],6),flush=True)
            if key>best[0]:best=(key,config,mix)
    _,config,mix=best
    b.dump(OUT/'trials.json',trials);b.dump(OUT/'selected.json',config);np.savez(OUT/'selected_probabilities.npz',**mix)
    rows=[r for s in dur for r in v4.decode_rows(mix[s],s,dur[s],config)]
    full=b.score_save('validation',rows,dur,gt,OUT,ev)
    diag=b.score_save('diagnostic',[r for r in rows if r['stream_id'] in dd],dd,gt,OUT,ev)
    old=json.loads((HERE/'search_v4/selected_validation_report.json').read_text());old_d=json.loads((HERE/'search_v4/selected_diagnostic_report.json').read_text())
    promoted=b.ranking_key(full)>b.ranking_key(old) and diag['ranking_score']['f1_mean']>=old_d['ranking_score']['f1_mean']
    decision=dict(selected=config['name'],promoted=promoted,previous=old['ranking_score'],candidate=full['ranking_score'],diagnostic=diag['ranking_score'],previous_diagnostic=old_d['ranking_score'],note='Nine predefined mixtures, decoder selection on val_00..02. Reused diagnostic, not an independent test. V4 is unchanged.')
    b.dump(OUT/'decision.json',decision);print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
