"""Prespecified seed-paired bootstrap and evidence report."""
import csv
import json
import pathlib
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]

def analyze(rows,controls):
    seeds=list(range(96161001,96161009))
    assert [r['seed'] for r in rows]==seeds
    idx=np.random.default_rng(96169999).integers(0,8,(20000,8))
    def boot(values):
        v=np.array(values,dtype=float)
        b=v[idx].mean(axis=1)
        return dict(mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),per_seed=v.tolist()),b
    def get(seed,d,method):
        hit=[r for r in controls if r['seed']==seed and r['displacement']==d and r['method']==method]
        assert len(hit)==1
        return hit[0]
    paired=[]
    for r in rows:
        s=r['seed']
        near=[get(s,d,'oracle') for d in [.38,.40]]
        ref=[get(s,d,'phase') for d in [.55,.70]]
        p=dict(seed=s,frozen_ce=r['final']['ce'],frozen_accuracy=1-r['final']['bler'],
               near_oracle_ce=float(np.mean([x['final']['ce'] for x in near])),
               near_oracle_accuracy=float(np.mean([1-x['final']['bler'] for x in near])),
               reference_phase64_ce=float(np.mean([x['final']['ce'] for x in ref])),
               reference_phase64_accuracy=float(np.mean([1-x['final']['bler'] for x in ref])))
        p['frozen_minus_near_ce']=p['frozen_ce']-p['near_oracle_ce']
        p['frozen_minus_reference_ce']=p['frozen_ce']-p['reference_phase64_ce']
        for d,m in [(.38,'oracle'),(.40,'oracle'),(.55,'phase'),(.70,'phase')]:
            rr=get(s,d,m)
            p[f'{m}_{d:.2f}_ce']=rr['final']['ce']
            p[f'{m}_{d:.2f}_accuracy']=1-rr['final']['bler']
            p[f'frozen_minus_{m}_{d:.2f}_ce']=p['frozen_ce']-rr['final']['ce']
        paired.append(p)
    with (ROOT/'results/PAIRED_BY_SEED.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    summaries={};samples={}
    for name in paired[0]:
        if name=='seed':continue
        summaries[name],samples[name]=boot([r[name] for r in paired])
    primary=summaries['frozen_minus_near_ce'];lo,hi=primary['ci95'];mean=primary['mean']
    if lo>=-.02 and hi<=.02:
        verdict='Proposition F is supported under the ±0.02 CE equivalence margin in this setting'
        code='support_equivalence'
        interpretation='The frozen versus near-zero oracle difference interval lies within the prespecified equivalence margin; this does not imply transmitter parameters contain no changes or information.'
    elif mean>.02 and (lo>0 or hi<0):
        verdict='Proposition F is rejected in this setting: near-zero physical return still yields useful joint-training gains'
        code='reject_useful_near_return'
        interpretation='Frozen-transmitter CE is higher, with a difference exceeding the prespecified margin and excluding zero; near-zero performance cannot be attributed solely to receiver adaptation.'
    elif mean<-.02 and (lo>0 or hi<0):
        verdict='Proposition F is rejected in this setting: near-zero return worsens joint training'
        code='reject_harmful_near_return'
        interpretation='Frozen-transmitter CE is lower, with a difference beyond the negative prespecified margin and excluding zero; the training methods cannot be called equivalent.'
    else:
        verdict='Insufficient evidence: the experiment supports neither equivalence nor the rejection criterion for proposition F'
        code='insufficient_evidence'
        interpretation='The paired interval from 8 seeds satisfies neither prespecified criterion; no grid expansion or additional seeds are used.'
    out=dict(verdict=verdict,verdict_code=code,primary=primary,
             transmitter_total_contribution=summaries['frozen_minus_reference_ce'],
             summaries=summaries,bootstrap_replicates=20000,bootstrap_seed=96169999,
             independent_unit='seed, average displacements within seed first',seeds=seeds)
    (ROOT/'results/PRIMARY_RESULTS.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    np.savez_compressed(ROOT/'results/bootstrap_raw.npz',indices=idx,**samples)
    def fmt(key):
        r=summaries[key];return f"{r['mean']:.6f} [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]"
    def avg(key):return summaries[key]['mean']
    lines=['# Proposition F: frozen transmitter ablation','',f'**{verdict}.**','',interpretation,'',
           'The primary endpoint is frozen minus near-zero oracle CE, averaging 0.38 and 0.40 within each seed first.',
           f"Difference **{fmt('frozen_minus_near_ce')}** (mean and paired bootstrap 95% interval, nats/block).",
           'The prespecified equivalence margin is [-0.02,0.02]; rejection requires the mean to exceed ±0.02 and the interval to exclude zero.','',
           '| Condition | Mean test CE | Mean accuracy |','|---|---:|---:|',
           f"| frozen_tx | {avg('frozen_ce'):.6f} | {100*avg('frozen_accuracy'):.3f}% |",
           f"| Near-zero oracle, within-seed mean at 0.38/0.40 | {avg('near_oracle_ce'):.6f} | {100*avg('near_oracle_accuracy'):.3f}% |",
           f"| Reference phase64, within-seed mean at 0.55/0.70 | {avg('reference_phase64_ce'):.6f} | {100*avg('reference_phase64_accuracy'):.3f}% |",'',
           f"Total contribution of transmitter learning: CE(frozen_tx)−CE(reference phase64) = **{fmt('frozen_minus_reference_ce')}**.",'',
           '| Displacement and reused control | Control CE | Control accuracy | frozen_tx − control CE, 95% interval |',
           '|---|---:|---:|---|']
    for d,m in [(.38,'oracle'),(.40,'oracle'),(.55,'phase'),(.70,'phase')]:
        key=f'{m}_{d:.2f}'
        lines.append(f"| {d:.2f} / {'phase64' if m=='phase' else m} | {avg(key+'_ce'):.6f} | {100*avg(key+'_accuracy'):.3f}% | {fmt('frozen_minus_'+key+'_ce')} |")
    lines+=['','| Seed | frozen CE | near-zero oracle CE | reference phase64 CE | primary difference |',
            '|---|---:|---:|---:|---:|']
    for p in paired:
        lines.append(f"| {p['seed']} | {p['frozen_ce']:.6f} | {p['near_oracle_ce']:.6f} | {p['reference_phase64_ce']:.6f} | {p['frozen_minus_near_ce']:.6f} |")
    lines+=['','## Execution and pairing checks','',
      '- Exactly 8 seeds, trained once each: 4032 training blocks and 126 receiver Adam updates; the codebook never changes. No controls were retrained.',
      '- The 32 historical controls come from the same E2 result file, uniquely selected by seed, Nr, spectrum, displacement, method and pilot variant.',
      '- Initial codebooks and forward channels match the controls elementwise; noise and labels use the same streams and draw order. All 32 saved models reproduce test metrics (CE error <1e-12 and identical error counts).',
      '- frozen_tx has no displacement parameter, return link or calibration; its data do not depend on displacement.',
      '- The bootstrap resamples 8 seeds 20000 times with seed 96169999; individual displacement results are supplementary.','',
      '## Limitations','',
      '- These results use 8 seeds and a fixed configuration and budget; they do not generalize automatically to other SNRs, receivers, training durations or antenna sizes.',
      '- Joint-training gains include receiver readaptation to the codebook trajectory; they do not isolate the direct transmitter contribution or quantify a receiver-only share.',
      '- Historical per-sample channel and noise arrays were not saved. Reconstruction uses deterministic source, seeds and shapes, and is checked against saved model metrics.',
      '- All controls use the same E2 result file; no CSI arm or other training was added.','',
      '## Evidence','',
      '- [Freeze manifest](freeze_manifest.json), [primary results](results/PRIMARY_RESULTS.json), [paired results](results/PAIRED_BY_SEED.csv), [pairing audit](results/PAIRING_AUDIT.json).',
      '- [Frozen results](results/all_runs.json), [reused controls](reference/selected_controls.json), [training source](src/frozen_tx.py), [analysis source](src/analyze.py).',
      '- results/data_<seed>.npz contains training/test data; model_<seed>.npz contains final model and Adam state; trace_<seed>.csv contains 126 updates.',
      '- results/bootstrap_raw.npz contains bootstrap indices and resampled means.','']
    (ROOT/'VERDICT.md').write_text('\n'.join(lines))
    print(json.dumps({k:out[k] for k in ['verdict','primary','transmitter_total_contribution']},ensure_ascii=False),flush=True)
