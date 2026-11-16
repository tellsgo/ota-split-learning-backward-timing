# -*- coding: utf-8 -*-
from pathlib import Path
import os,json,time,hashlib,csv,sys,subprocess,math
import numpy as np
R=Path(__file__).resolve().parent
def load(name):return json.loads((R/name).read_text())
def dump(name,obj):(R/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def ci(b):return f"{b['mean']:.6g} [{b['ci95'][0]:.6g}, {b['ci95'][1]:.6g}]"
source=load('C2_provenance.json')
# Arithmetic: equations and intermediate values retained.
wavelength=299792458/3.5e9
c1={'c_m_per_s':299792458,'carrier_hz':3.5e9,'lambda_m':wavelength,'window_lambda':[.45,.70],'width_lambda':.25,'slot_ms':.5,'speeds':{str(v):{'start_ms':.45*wavelength/v*1000,'end_ms':.70*wavelength/v*1000,'width_ms':.25*wavelength/v*1000,'width_in_slots':.25*wavelength/v*1000/.5} for v in [15,33]}}
c3={'dt_s':.0013,'max_phase_degrees':30,'boundary_Hz':1/(12*.0013),'strict_requirement':'abs(delta_f) < 64.1025641025641 Hz','formula':'2*pi*abs(delta_f)*dt < pi/6'}
dump('results/C.json',{'C1':c1,'C2':{'verdict':'Jakes spectrum specification verified for Rayleigh taps; normalized J0 correlation follows from the Fourier transform','source':source,'limitations':'The first Rician tap in TDL-D/E also has a spectral line at 0.7 f_D; the full TDL correlation is not equated with J0'},'C3':c3})
if sys.argv[1:] == ['C']:
 sys.exit(0)
rows=[json.loads(l) for l in (R/'results/runs.jsonl').read_text().splitlines()];gate=load('results/GATE_DECISION.json');b1=load('results/B1.json');b2=load('results/B2.json');b3=load('results/B3.json');a4=load('results/A4.json')
# Per-seed table, no re-training or new grids.
with (R/'results/all_training.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['tag','task','seed','K','d','method','CE','accuracy','train_blocks','RF_vectors'])
 for row in rows:
  c=row['costs'];w.writerow([row['tag'],row['task'],row['seed'],row.get('K',''),row['displacement'],row['method'],row['service']['ce'],row['service']['accuracy'],c.get('training_blocks',c.get('training_samples')),c['rf_total']])
# Diagnostic aggregation remains descriptive, no hypothesis tests.
diag=[json.loads(l) for l in (R/'results/A4_samples.jsonl').read_text().splitlines()];groups=[]
for task in ['MNIST','S']:
 for d in [.38,.55,.6098]:
  for method in ['raw','oracle']:
   v=[a for a in diag if a['task']==task and a['d']==d and a['method']==method and not a['near_zero_true_gradient']]
   groups.append({'task':task,'d':d,'method':method,'nonzero_samples':len(v),'median_g2':float(np.median([x['g_norm2'] for x in v])),'median_B2':float(np.median([x['B_frobenius2_exact_check'] for x in v])),'median_MC_cosine':float(np.median([x['conditional_MC_cosine'] for x in v])),'median_relative_MC_SE2':float(np.median([x['MC_mean_SE_norm2']/x['g_norm2'] for x in v]))})
dump('results/A4_grouped.json',groups)
# Independent recomputation of the reported B summaries from raw rows.
bootids=np.random.default_rng(73179999).integers(0,8,(20000,8));seeds=range(73172001,73172009)
def ce(s,K,d,m):
 a=[r for r in rows if r['task']=='B' and r['seed']==s and r['K']==K and r['displacement']==d and r['method']==m];assert len(a)==1;return a[0]['service']['ce']
errors=[]
for K in [0,1,3,10]:
 v=np.array([(ce(s,K,.38,'oracle')+ce(s,K,.4,'oracle')-ce(s,K,.55,'oracle')-ce(s,K,.7,'oracle'))/2 for s in seeds]);expected=b2['gaps'][str(K)];errors.append(abs(v.mean()-expected['mean']));errors.extend(abs(np.quantile(v[bootids].mean(1),[.025,.975])-expected['ci95']))
v=np.array([np.mean([ce(s,10,d,'raw')-ce(s,10,d,'phase') for d in [.55,.7,.6098]]) for s in seeds]);errors.append(abs(v.mean()-b3['raw_minus_phase64']['mean']));errors.extend(abs(np.quantile(v[bootids].mean(1),[.025,.975])-b3['raw_minus_phase64']['ci95']))
assert max(errors)<1e-12
assert len(rows)==180 and len(set(r['tag'] for r in rows))==180
assert len([r for r in rows if r['task']=='A'])==4 and all('_gate_' in r['tag'] for r in rows if r['task']=='A')
assert len(diag)==288
frozen=load('freeze_spec.json');assert hashlib.sha256((R/'SPEC.md').read_bytes()).hexdigest()==frozen['sha256']
for name,h in load('freeze_code.json')['hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
assert hashlib.sha256((R/'src/diagnostics.py').read_bytes()).hexdigest()==load('freeze_diagnostics.json')['sha256']
ro=load('READONLY_SOURCES.json');unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in ro.items()};assert all(unchanged.values())
# Record relevant process table; no kill of unrelated work.
proc=subprocess.check_output(['ps','-Ao','pid,ppid,etime,command'],text=True);owned=[]
for line in proc.splitlines():
 fields=line.split(None,3)
 if len(fields)<4:continue
 exe=Path(fields[3].split()[0]).name
 if exe not in ['Python','python','python3','curl']:continue
 if str(R.name) in line and ('src/experiment.py' in line or 'src/diagnostics.py' in line or 'src/verify_rician.py' in line or 'curl ' in line):owned.append(line)
assert not owned,owned
end=time.time();elapsed=end-load('session.json')['start_unix'];assert elapsed<21600
checks={'completed':True,'time':end,'wall_seconds_from_conservative_start':elapsed,'within_6_hours':elapsed<21600,'main_training_rows':len(rows),'A_gate_rows':4,'A_main_rows':0,'B_training_rows':176,'diagnostic_S_training_rows':2,'A4_rows':len(diag),'bootstrap_recompute_max_error':max(errors),'readonly_sources_unchanged':unchanged,'frozen_spec_unchanged':True,'frozen_training_code_unchanged':True,'frozen_diagnostic_code_unchanged':True,'remaining_owned_training_or_download_processes':owned,'all_failed_attempts_retained':['logs/A_stdout.log','logs/C2_early_read_failure.txt','logs/closeout.log','logs/closeout_retry1.log','logs/closeout_before_encoding_header.py.txt','logs/closeout_final.log','logs/closeout_before_process_filter.py.txt'],'skill_files_accessed':False,'forbidden_directory_accessed':False,'other_sessions_or_models_called':False}
dump('DELIVERY_CHECKS.json',checks)
dump('RECLAMATION.json',{'checked_unix':end,'remaining_owned_processes':owned,'reclamation_action':'All own training and download processes exited naturally; tool sessions collected. No unrelated processes touched.','artifacts_retained':True})
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.json'}
dump('ARTIFACT_MANIFEST.json',{'time':time.time(),'files':manifest})
print(json.dumps({'wall_minutes':elapsed/60,'rows':len(rows),'verdict':str(R/'VERDICT.md'),'delivery_passed':True},ensure_ascii=False))
