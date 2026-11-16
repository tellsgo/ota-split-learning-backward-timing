from core import *
import sys,traceback,platform
from statistics_a import summary
GRID=[.35,.4,.45,.55,.7,.85,.9,1.05,1.2,1.35,1.4,1.45]
def gate():
 rows=[];steps=[];chosen=None
 for level,(var,msg) in enumerate([(.1,8),(.05,8),(.1,4)],1):
  configure({'forward_noise_variance':var,'messages':msg});local=[]
  for s in GATE:
   row=train('message',s,geo('isotropic',0),'csi');row['ladder_level']=level;rows.append(row);local.append(row['service']['accuracy']);print('GATE',level,s,row['service'],flush=True)
  step={'level':level,'config':CONFIG.copy(),'accuracies':local,'mean_accuracy':float(np.mean(local)),'passed':bool(np.mean(local)>=.9)};steps.append(step)
  dump(R/'gate/results/partial_runs.json',rows)
  if step['passed']:chosen=step;break
 if chosen is None:chosen=max(steps,key=lambda x:x['mean_accuracy'])
 dump(R/'gate/results/all_runs.json',rows);dump(R/'gate/results/decision.json',{'steps':steps,'selected':chosen,'selection_time':time.time(),'gate_passed':chosen['passed'],'unrun_levels':[i for i in [1,2,3] if i not in [x['level'] for x in steps]]})
def summary_b(rows):
 def get(s,nr,d,m):
  x=[r['service']['ce'] for r in rows if r['seed']==s and r['nr']==nr and r['geometry']['displacement_lambda']==d and r['method']==m];assert len(x)==1;return x[0]
 gaps={}
 for nr,ms in [(1,['oracle','phase','csi']),(3,['oracle','polar_oracle','polar64','csi'])]:
  for m in ms:gaps[f'nr{nr}_{m}']=[np.mean([get(s,nr,d,m) for d in [.38,.4]])-np.mean([get(s,nr,d,m) for d in [.55,.7]]) for s in SEEDS]
 a=boot(gaps['nr1_oracle']);b=boot(gaps['nr3_polar_oracle']);c=boot(np.array(gaps['nr1_oracle'])-gaps['nr3_polar_oracle'])
 passed={'B1':a['mean']>=.05 and a['ci95'][0]>0,'B2':b['ci95'][1]<.02,'B3':c['mean']>=.05 and c['ci95'][0]>0}
 return {'B1':a,'B2':b,'B3':c,'passed':passed,'verdict':{k:'Supported in this setting' if v else 'Falsification threshold reached (prespecified criterion not met)' for k,v in passed.items()},'all_gap_diagnostics':{k:boot(v) for k,v in gaps.items()},'paired_unit':'seed; average positions within near/reference first','B2_bound':.02}
def run(e):
 p=R/e;t=time.time();verify_freeze(e);dump(p/'results/preflight.json',verify())
 try:
  if e=='gate':gate()
  else:
   decision=json.loads((R/'gate/results/decision.json').read_text());configure(decision['selected']['config']);rows=[]
   if e=='E2':cases=[('isotropic',d) for d in [.38,.4,.55,.7]];arms=[('phase',64),('phase',256),('phase',1024),('oracle',0),('raw',64),('csi',64)]
   elif e=='E3':cases=[('isotropic',d) for d in [.38,.4,.42,D1]];arms=[(m,64) for m in ['raw','phase','phase2','phase1','scalar','csi']]
   elif e=='E1':cases=[('isotropic',d) for d in GRID];arms=[(m,64) for m in ['raw','phase','csi']]
   elif e=='E4':cases=[('isotropic',d) for d in [.55,.7,1.05,1.2]]+[('broad',d) for d in [.1,.2,.3,.4,.5]];arms=[(m,64) for m in ['phase','phase2','phase1','scalar']]
   elif e=='B':cases=[('isotropic',d) for d in [.38,.4,.55,.7]]
   for sp,d in cases:
    for s in SEEDS:
     if e=='B':
      for nr,ms in [(1,['oracle','phase','csi']),(3,['oracle','polar_oracle','polar64','csi'])]:
       for m in ms:rows.append(train('message',s,geo(sp,d,nr=nr),m,64))
     else:
      for m,k in ([('phase',64),('csi',64)] if e=='E3' and d==D1 else arms):rows.append(train('message',s,geo(sp,d),m,k))
    dump(p/'results/partial_runs.json',rows);print(e,sp,d,len(rows),'seconds',time.time()-t,flush=True)
   dump(p/'results/all_runs.json',rows);out=summary_b(rows) if e=='B' else summary(e,rows)
   if e=='E2':
    out.pop('stronger_than_array_0p25',None);out.pop('cross_task_comparison_caveat',None);out['array_comparison']='Use paired B experiment in this run; no cross-task historical CE comparison.'
   dump(p/'results/primary.json',out)
   stop=not all(out['passed'].values()) if e=='B' else out['verdict']!='Supported in this setting'
   if stop:dump(p/'results/STOP.json',{'reason':out['verdict'],'action':'Frozen primary grid complete. Stop failed line; no extra tuning or grid. Continue other prespecified independent tests.','time':time.time()})
 except BaseException:
  (p/'results/failure.txt').write_text(traceback.format_exc());raise
 dump(p/'results/runtime.json',{'host':platform.node(),'pid':os.getpid(),'wall_seconds':time.time()-t,'experiment':e});print('DONE',e,flush=True)
if __name__=='__main__':run(sys.argv[1])
