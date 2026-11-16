"""One receiver-only training per seed, paired to frozen historical records."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import csv
import hashlib
import json
import pathlib
import platform
import resource
import signal
import sys
import time
import traceback
import numpy as np
import scipy
import communication as M
import core as historical

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEEDS = list(range(96161001, 96161009))
CONDITIONS = [(0.38, 'oracle', 0), (0.40, 'oracle', 0),
              (0.55, 'phase', 64), (0.70, 'phase', 64)]
N = 4032

def dump(name, obj):
    (ROOT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def digest(a):
    return hashlib.sha256(np.require(a, requirements=['C']).tobytes()).hexdigest()

def unpack(p):
    return np.asarray(p['real']) + 1j*np.asarray(p['imag'])

def verify_freeze():
    man = json.loads((ROOT/'freeze_manifest.json').read_text())
    for name, expected in man['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    return man

def select_reference():
    all_rows = json.loads((ROOT/'reference/all_runs.json').read_text())
    rows = []
    for seed in SEEDS:
        for d, method, pilots in CONDITIONS:
            hits = [r for r in all_rows if r['seed']==seed and r['nr']==1
                    and r['geometry']['spectrum']=='isotropic'
                    and r['displacement']==d and r['method']==method
                    and r['pilot_variant']==pilots]
            assert len(hits)==1, (seed,d,method,len(hits))
            row = hits[0]
            assert row['updates']==126
            assert row['config']=={'messages':8,'forward_noise_variance':0.1}
            assert row['costs']['training_samples']==N
            rows.append(row)
    dump('reference/selected_controls.json', rows)
    return rows

def make_data(seed):
    z = M.norm(M.cn(M.rng(seed,101),(8,2,4)))
    # Identical first draw to historical.channels for Nr=1; no displacement needed.
    f = M.cn(M.rng(seed,103),(N,1,2))
    noise = np.sqrt(.1)*M.cn(M.rng(seed,104),(N,1,4))
    lab = M.labels(seed,N)
    tf = M.cn(M.rng(seed,107),(8192,1,2))
    tn = np.sqrt(.1)*M.cn(M.rng(seed,109),(8192,1,4))
    tl = np.tile(np.arange(8),1024)
    return dict(z0=z,train_channel=f,train_noise=noise,train_labels=lab,
                test_channel=tf,test_noise=tn,test_labels=tl)

def audit_data(seed, data, controls):
    z0 = data['z0']
    audit = {'seed':seed,'array_sha256':{k:digest(v) for k,v in data.items()},'controls':[]}
    # Reconstruct the exact original noise draw order, including subsequent unused draws.
    rn = M.rng(seed,104)
    original_noise = np.sqrt(M.C['forward_noise_variance'])*M.cn(rn,(N,M.C['nr'],4))
    _reverse_noise = .03*M.cn(rn,(N,2,4))
    _pilot_noise = np.sqrt(M.C['forward_noise_variance'])*M.cn(rn,(N,M.C['nr'],2))
    assert np.array_equal(original_noise,data['train_noise'])
    assert np.array_equal(data['train_labels'],M.labels(seed,N))
    assert np.array_equal(data['test_noise'],np.sqrt(.1)*M.cn(M.rng(seed,109),(8192,1,4)))
    for row in controls:
        if row['seed'] != seed: continue
        d = row['displacement']
        g = historical.geo('isotropic',d,nr=1)
        f,_ = historical.channels(g,M.rng(seed,103),N)
        tf,_ = historical.channels(g,M.rng(seed,107),8192)
        assert np.array_equal(z0,unpack(row['initial_codebook'])), (seed,d,'initial_codebook')
        assert np.array_equal(f,data['train_channel']), (seed,d,'forward_channel')
        assert np.array_equal(tf,data['test_channel']), (seed,d,'test_channel')
        # Re-evaluation of saved models only: never call historical.train / M.train.
        final = M.evaluate(unpack(row['final_codebook']),np.array(row['receiver_weights']),
                           np.array(row['receiver_bias']),tf,data['test_noise'],data['test_labels'])
        err = abs(final['ce']-row['final']['ce'])
        assert err<1e-12 and final['errors']==row['final']['errors'], (seed,d,err)
        assert final['bler']==row['final']['bler']
        assert row['service']['ce']==row['final']['ce']
        audit['controls'].append(dict(displacement=d,method=row['method'],initial_codebook_exact=True,
          train_channel_exact=True,test_channel_exact=True,test_ce_error=err,test_errors_exact=True))
    return audit

def train(seed, data):
    z = data['z0'].copy()
    z.setflags(write=False)
    W = np.zeros((8,16)); bias = np.zeros(8)
    aw = M.Adam(W.shape); ab = M.Adam(bias.shape)
    trace = []
    for k in range(0,N,32):
        sl = slice(k,k+32); ll = data['train_labels'][sl]
        x = M.norm(z[ll])
        y = np.einsum('bri,bit->brt',data['train_channel'][sl],x)+data['train_noise'][sl]
        feat = M.features(y)
        loss,p,pred = M.decode(feat,W,bias,ll)
        gw = M.emat(p.T,feat)/32 + .0002*W
        gb = p.mean(axis=0)
        W = np.clip(aw.step(W,gw,.03),-5,5)
        bias = np.clip(ab.step(bias,gb,.03),-5,5)
        assert np.all(np.isfinite(W)) and np.all(np.isfinite(bias))
        assert np.array_equal(z,data['z0'])
        trace.append(dict(seed=seed,update=k//32+1,blocks=k+32,pre_update_train_ce=float(loss.mean()),
                          pre_update_train_accuracy=float(np.mean(pred==ll)),
                          W_norm=float(np.linalg.norm(W)),bias_norm=float(np.linalg.norm(bias)),
                          codebook_max_abs_change=float(np.max(np.abs(z-data['z0'])))))
    assert aw.t==ab.t==126
    final = M.evaluate(z,W,bias,data['test_channel'],data['test_noise'],data['test_labels'])
    initial = M.evaluate(z,np.zeros_like(W),np.zeros_like(bias),data['test_channel'],data['test_noise'],data['test_labels'])
    np.savez_compressed(ROOT/f'results/model_{seed}.npz',z0=data['z0'],z_final=z,W=W,bias=bias,
                        adam_w_m=aw.m,adam_w_v=aw.v,adam_bias_m=ab.m,adam_bias_v=ab.v)
    with (ROOT/f'results/trace_{seed}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(trace[0]));writer.writeheader();writer.writerows(trace)
    return dict(seed=seed,method='frozen_tx',training_blocks=N,updates=126,initial=initial,final=final,
                initial_codebook=M.pack(data['z0']),final_codebook=M.pack(z),
                receiver_weights=W.tolist(),receiver_bias=bias.tolist(),
                checks={'codebook_exactly_unchanged':True,'max_abs_codebook_change':0.0,
                        'transmitter_updates':0,'reverse_channel_uses':0,'displacement_parameter':None})

def main():
    start=time.time()
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));signal.alarm(1800)
    verify_freeze()
    M.C.update(nr=1,messages=8,forward_noise_variance=.1)
    # Confirm reused source identity against the historical frozen E2 execution.
    old=json.loads((ROOT/'reference/historical_freeze_manifest.json').read_text())['hashes']
    for name in ['communication.py','core.py','reference_geo.py']:
        assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==old['shared/src/'+name]
    dump('ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
                                host=platform.node(),platform=platform.platform(),pid=os.getpid(),started=start))
    controls=select_reference();rows=[];audits=[]
    for seed in SEEDS:
        data=make_data(seed)
        audits.append(audit_data(seed,data,controls))
        np.savez_compressed(ROOT/f'results/data_{seed}.npz',**data)
        dump('results/PAIRING_AUDIT.json',audits)
        row=train(seed,data);rows.append(row)
        dump('results/partial_runs.json',rows)
        print('FINISHED',seed,json.dumps(row['final']), 'elapsed',round(time.time()-start,3),flush=True)
    dump('results/all_runs.json',rows)
    from analyze import analyze
    analyze(rows,controls)
    verify_freeze()
    dump('results/runtime.json',dict(wall_seconds=time.time()-start,seeds_completed=len(rows),pid=os.getpid(),host=platform.node()))
    print('DONE: all 8 seeds, 32 saved-model checks, paired bootstrap and report',flush=True)

if __name__=='__main__':
    try: main()
    except BaseException:
        (ROOT/'results/failure.txt').write_text(traceback.format_exc())
        raise
