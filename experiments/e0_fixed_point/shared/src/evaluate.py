import siso as S
from siso import np,R,SEEDS,dump,geo,train,boot,D1
import time,sys,traceback,platform,os,hashlib
GRID=[.35,.4,.45,.55,.7,.85,.9,1.05,1.2,1.35,1.4,1.45]
def ce(rows,s,d,m,k=None,sp='isotropic'):
 a=[r['service']['ce'] for r in rows if r['seed']==s and r['geometry']['displacement_lambda']==d and r['method']==m and r['geometry']['spectrum']==sp and (k is None or r['pilot_variant']==k)]
 assert len(a)==1,(s,d,m,k,sp,len(a));return a[0]
def summary(e,rows):
 if e=='E1':
  neg=[.4,.45,.55,.7,.85];pos=[.9,1.05,1.2,1.35]
  v=[np.mean([ce(rows,s,d,'raw') for d in neg])-np.mean([ce(rows,s,d,'raw') for d in pos]) for s in SEEDS];b=boot(v)
  return {'negative_lobe':neg,'second_positive_lobe':pos,'raw_CE_improvement':b,'verdict':'supported under this setting' if b['mean']>=.02 and b['ci95'][0]>0 else 'falsification threshold met','raw_minus_csi_negative':boot([np.mean([ce(rows,s,d,'raw')-ce(rows,s,d,'csi') for d in neg]) for s in SEEDS]),'raw_minus_csi_positive':boot([np.mean([ce(rows,s,d,'raw')-ce(rows,s,d,'csi') for d in pos]) for s in SEEDS])}
 if e=='E2':
  near=[.38,.4];far=[.55,.7]
  gap=np.array([np.mean([ce(rows,s,d,'oracle') for d in near])-np.mean([ce(rows,s,d,'oracle') for d in far]) for s in SEEDS])
  gain=np.array([np.mean([ce(rows,s,d,'phase',64)-ce(rows,s,d,'phase',1024) for d in near]) for s in SEEDS])
  ids=np.random.default_rng(95159999).integers(0,8,(20000,8));bg=gap[ids].mean(1);bi=gain[ids].mean(1);ratio=bi/bg;gr=boot(gap)
  rat={'mean':float(gain.mean()/gap.mean()),'ci95':np.quantile(ratio,[.025,.975]).tolist(),'nonpositive_bootstrap_denominator':int(sum(bg<=0))}
  false=gr['mean']<.02 or rat['mean']>=.5;support=gr['ci95'][0]>.02 and rat['ci95'][1]<.5 and min(bg)>0
  return {'oracle_near_minus_reference':gr,'pilot_improvement_64_to_1024':boot(gain),'fraction_gap_closed':rat,'verdict':'falsification threshold met' if false else 'supported under this setting' if support else 'insufficient evidence','stronger_than_array_0p25':gr['mean']>.25,'cross_task_comparison_caveat':'array benchmark uses binary geometry CE; present message CE is 8-class. Numeric greater-than is descriptive, not controlled causal evidence.'}
 if e=='E3':
  near=[.38,.4,.42];methods=['phase','phase2','phase1','scalar']
  best={str(d):min(methods,key=lambda m:np.mean([ce(rows,s,d,m) for s in SEEDS])) for d in near}
  v=[np.mean([ce(rows,s,d,best[str(d)])-ce(rows,s,D1,'phase') for d in near]) for s in SEEDS];b=boot(v)
  return {'best_method_per_d':best,'delay_improvement':b,'verdict':'supported under this setting' if b['mean']>=.02 and b['ci95'][0]>0 else 'falsification threshold met','selection_uncertainty_included':False,'per_d':{str(d):boot([ce(rows,s,d,best[str(d)])-ce(rows,s,D1,'phase') for s in SEEDS]) for d in near},'d1':D1}
 if e=='E4':
  iso=[.55,.7,1.05,1.2];br=[.1,.2,.3,.4,.5]
  def delta(sp,ds,m):return boot([np.mean([ce(rows,s,d,m,sp=sp)-ce(rows,s,d,'phase',sp=sp) for d in ds]) for s in SEEDS])
  a=delta('isotropic',iso,'phase1');b=delta('broad',br,'phase2');near90=[d for d in br if 60<=abs(geo('broad',d)['info']['phase_degrees'])<=120];c=delta('broad',near90,'phase1') if near90 else None
  return {'isotropic_1bit_minus_continuous':a,'broad_2bit_minus_continuous':b,'near90_displacements':near90,'broad_near90_1bit_minus_continuous':c,'isotropic_noninferior':a['ci95'][1]<.02,'broad_2bit_noninferior':b['ci95'][1]<.02,'broad_1bit_degraded':c is not None and c['mean']>=.02 and c['ci95'][0]>0,'verdict':'supported under this setting' if a['ci95'][1]<.02 and b['ci95'][1]<.02 and c and c['mean']>=.02 and c['ci95'][0]>0 else 'not all criteria met (see individual criteria)','per_d':{sp:{str(d):{m:delta(sp,[d],m) for m in ['phase1','phase2','scalar']} for d in ds} for sp,ds in [('isotropic',iso),('broad',br)]}}
def e0(p):
 rows=[];arrays={};n=65536;blocks=128
 def cosine(x,y):return np.vdot(x,y).real/(np.linalg.norm(x)*np.linalg.norm(y))
 for sp in ['isotropic','broad']:
  for direction in ['parallel','perpendicular']:
   for d in [0.,.1,.3,.5]:
    g=geo(sp,d,direction);target=float(np.cos(np.angle(g['alpha'])))
    for s in SEEDS:
     rr=S.rng(s,600);f=S.K.cn(rr,(1,2));seed=S.K.cn(rr,(1,))[0];truth=f.conj().ravel()*seed
     b=g['alpha']*f+g['qr'][0,0]*S.K.cn(rr,(n,2));v=b.conj()*seed;bm=v.reshape(blocks,n//blocks,2).mean(1);mu=bm.mean(0)
     obs=cosine(truth,mu);loo=np.array([cosine(truth,(blocks*mu-x)/(blocks-1)) for x in bm]);se=float(np.sqrt((blocks-1)/blocks*np.sum((loo-loo.mean())**2)))
     analytic=cosine(truth,g['alpha'].conjugate()*truth)
     arrays[f'{sp}_{direction}_{d}_{s}']=bm
     rows.append({'seed':s,'spectrum':sp,'direction':direction,'d':d,'target':target,'observed':float(obs),'mc_se':se,'analytic':float(analytic),'hf':S.pack(f),'seed_vector':S.pack(seed),'conditional_draws':n})
    print('E0',sp,direction,d,flush=True)
 dump(p/'results/all_runs.json',rows);np.savez_compressed(p/'results/conditional_block_means.npz',**arrays)
 cells=[]
 for sp in ['isotropic','broad']:
  for direction in ['parallel','perpendicular']:
   for d in [0.,.1,.3,.5]:
    rr=[r for r in rows if (r['spectrum'],r['direction'],r['d'])==(sp,direction,d)];diff=np.mean([r['observed']-r['target'] for r in rr]);se=np.sqrt(sum(r['mc_se']**2 for r in rr))/8
    z=abs(diff)/se if se>1e-14 else 0. if abs(diff)<1e-12 else 1e300
    cells.append({'spectrum':sp,'direction':direction,'d':d,'target':rr[0]['target'],'observed':float(np.mean([r['observed'] for r in rr])),'bias':float(diff),'mc_se':float(se),'z':float(z),'passed':bool(z<=5)})
 dump(p/'results/primary.json',{'cells':cells,'max_z':max(x['z'] for x in cells),'analytic_max_error':max(abs(r['analytic']-r['target']) for r in rows),'verdict':'supported under this setting' if all(c['passed'] for c in cells) else 'falsification threshold met','estimator':'cosine of conditional mean, never mean of individual cosines; jackknife MC SE over 128 independent blocks per seed'})
def main(e):
 p=R/e;t=time.time();man=__import__('json').loads((p/'freeze_manifest.json').read_text())
 for f,h in man['hashes'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
 dump(p/'results/preflight.json',S.verify())
 try:
  if e=='E0':e0(p)
  else:
   if e=='E2':cases=[('isotropic',d) for d in [.38,.4,.55,.7]];arms=[('phase',64),('phase',256),('phase',1024),('oracle',0),('raw',64),('csi',64)]
   elif e=='E3':cases=[('isotropic',d) for d in [.38,.4,.42,D1]];arms=[(m,64) for m in ['raw','phase','phase2','phase1','scalar','csi']]
   elif e=='E1':cases=[('isotropic',d) for d in GRID];arms=[(m,64) for m in ['raw','phase','csi']]
   elif e=='E4':cases=[('isotropic',d) for d in [.55,.7,1.05,1.2]]+[('broad',d) for d in [.1,.2,.3,.4,.5]];arms=[(m,64) for m in ['phase','phase2','phase1','scalar']]
   rows=[]
   for sp,d in cases:
    for s in SEEDS:
     for m,k in ([('phase',64),('csi',64)] if e=='E3' and d==D1 else arms):rows.append(train('message',s,geo(sp,d),m,k))
    dump(p/'results/partial_runs.json',rows);print(e,sp,d,len(rows),'seconds',time.time()-t,flush=True)
   dump(p/'results/all_runs.json',rows);out=summary(e,rows);dump(p/'results/primary.json',out)
   if out['verdict']!='supported under this setting':dump(p/'results/STOP.json',{'reason':out['verdict'],'action':'Frozen primary grid complete. Stop this line; no grid expansion or parameter tuning.','time':time.time()})
 except BaseException:
  (p/'results/failure.txt').write_text(traceback.format_exc());raise
 dump(p/'results/runtime.json',{'host':platform.node(),'pid':os.getpid(),'wall_seconds':time.time()-t,'experiment':e});print('DONE',e,flush=True)
if __name__=='__main__':main(sys.argv[1])
