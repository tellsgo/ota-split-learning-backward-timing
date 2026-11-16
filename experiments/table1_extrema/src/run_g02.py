import os, sys, pathlib, json, hashlib, datetime, time, traceback, resource, signal, platform
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parents[1]
for k in ['MPLCONFIGDIR','XDG_CACHE_HOME','TMPDIR']:os.environ[k]=str(R/'figures/cache')
import core as C
import numpy as np
from scipy.special import j0
CFG=json.loads((R/'config.json').read_text()); SEEDS=CFG['seeds']
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def verify_freeze():
 m=json.loads((R/'freeze_manifest.json').read_text())
 for p,h in m['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
 return True
def stats(v):
 v=np.asarray(v);ids=np.random.default_rng(CFG['bootstrap_seed']).integers(0,len(v),(CFG['bootstrap_replicates'],len(v)))
 return {'mean':float(v.mean()),'ci95':np.quantile(v[ids].mean(1),[.025,.975]).tolist(),'per_seed':v.tolist()}
def values(rows,d,m):
 a=[]
 for s in SEEDS:
  rr=[r for r in rows if r['seed']==s and r['displacement']==d and r['arm']==m];assert len(rr)==1
  a.append(rr[0]['service']['ce'])
 return np.array(a)
def evaluate_claim(rows,d):
 a=values(rows,d,'phase_64'); b=values(rows,d,'CSI'); c=stats(a-b)
 if d==.610:
  yes=c['mean']<=.02; verdict='Supported under this setting' if yes else 'Falsification threshold reached'
  return 'G2.1',{'verdict':verdict,'phase_minus_CSI':c}
 ref=stats(a-values(rows,.610,'phase_64'))
 if d==1.117:
  if c['mean']<=.02 and c['ci95'][1]<=.025:verdict='Falsification threshold reached'
  elif c['mean']>.02 and ref['mean']>=.02:verdict='CE prediction supported under this setting'
  else:verdict='Not supported; no explicit falsification threshold reached'
  return 'G2.2',{'verdict':verdict,'phase_minus_CSI':c,'phase_minus_d1':ref}
 if ref['mean']<.02:verdict='Falsification threshold reached'
 elif ref['mean']>=.04:verdict='Quantitative CE prediction supported under this setting'
 else:verdict='Not supported; no explicit falsification threshold reached'
 return 'G2.3',{'verdict':verdict,'phase_minus_CSI':c,'phase_minus_d1':ref,'phase_minus_d2':stats(a-values(rows,1.117,'phase_64'))}
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));signal.alarm(1800)
 t=time.time();verify_freeze();C.configure({'messages':8,'forward_noise_variance':.1})
 check=C.verify();geometry=[]
 for d in CFG['displacements']:
  g=C.geo('isotropic',d);rho=g['alpha'];err=abs(rho-j0(2*np.pi*d));assert err<1e-12
  geometry.append({'d':d,'rho_real':rho.real,'rho_imag':rho.imag,'J0':float(j0(2*np.pi*d)),'J0_error':err,'effective_snr':abs(rho)**2/(1-abs(rho)**2)})
 check['geometry']=geometry;dump(R/'results/preflight.json',check)
 rows=[];claims={}
 for d in CFG['execution_order']:
  for s in SEEDS:
   for arm,m in CFG['methods'].items():
    row=C.train('message',s,C.geo('isotropic',d),m,64);row['arm']=arm;rows.append(row)
    with (R/'results/runs.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'completed':len(rows),'d':d,'seed':s,'arm':arm,'ce':row['service']['ce'],'elapsed_s':time.time()-t}),flush=True)
  dump(R/'results/partial_runs.json',rows)
  if d!=.383:
   name,result=evaluate_claim(rows,d);claims[name]=result;dump(R/'results/claims.json',claims)
   print(name,json.dumps(result,ensure_ascii=False),flush=True)
   if result['verdict']=='Falsification threshold reached':dump(R/'results'/('STOP_'+name+'.json'),{'claim':name,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':result,'action':'Stop extending this claim without additional grids, seeds, or tuning; continue the other prespecified independent claims and d0 diagnostics.'})
 dump(R/'results/all_runs.json',rows)
 summaries=[]
 for d in CFG['displacements']:
  for arm in CFG['methods']:
   r=[r for r in rows if r['displacement']==d and r['arm']==arm]
   summaries.append({'d':d,'arm':arm,'ce':stats(values(rows,d,arm)),'mean_accuracy':float(np.mean([x['service']['accuracy'] for x in r]))})
 dump(R/'results/summary.json',{'seeds':SEEDS,'claims':claims,'arms':summaries,'geometry':geometry})
 assert len(rows)==96
 csi_invariance=max(float(np.max(abs(values(rows,d,'CSI')-values(rows,.610,'CSI')))) for d in CFG['displacements'])
 assert csi_invariance==0
 for r in rows:
  assert r['updates']==(170 if r['arm']=='CSI' else 126)
  assert np.isfinite(r['service']['ce'])
  assert r['costs']['training_samples']==(5440 if r['arm']=='CSI' else 4032)
 verify_freeze()
 dump(R/'results/runtime.json',{'started_epoch':t,'finished_epoch':time.time(),'wall_seconds':time.time()-t,'pid':os.getpid(),'host':platform.node(),'python':sys.version,'numpy':np.__version__,'rows':len(rows),'csi_max_displacement_ce_difference':csi_invariance,'freeze_verified':True})
 print('DONE G02',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  (R/'results/failure.txt').write_text(traceback.format_exc());raise
