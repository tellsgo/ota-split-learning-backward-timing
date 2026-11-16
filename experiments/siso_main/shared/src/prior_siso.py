import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,pathlib,time,hashlib,platform,traceback
import numpy as np
from scipy.special import j0,jn_zeros
import learning_siso as K
import communication_siso as M
R=pathlib.Path(__file__).resolve().parents[2]
SEEDS=list(range(95151001,95151009)); GATE=[95150001,95150002]; D1=float(jn_zeros(1,1)[0]/(2*np.pi))
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def pack(z):return {'real':np.asarray(z).real.tolist(),'imag':np.asarray(z).imag.tolist()}
def geo(sp,d,direction='parallel'):
 # Exactly first receive port of reference experiment.geo; identical angular weights/sign.
 th=np.arange(128)*2*np.pi/128;mu=np.deg2rad(35)
 p=np.ones(128) if sp=='isotropic' else .7*np.exp(2*np.cos(th-mu))/np.i0(2)+.3
 p/=p.sum();u=np.cos(th) if direction=='parallel' else np.sin(th)
 rho=complex(sum(p*np.exp(2j*np.pi*d*u)))
 if d==0:rho=1+0j
 q=max(0.,1-abs(rho)**2)
 a=np.array([[rho]]);qr=np.array([[np.sqrt(q)]]);one=np.ones((1,1))
 return {'r':one,'r0':one,'t':a,'rt':a,'a':a,'q':np.array([[q]]),'root':one,'qr':qr,'alpha':rho,'d':d,'info':{'spectrum':sp,'direction':direction,'displacement_lambda':d,'rho':pack(rho),'magnitude':abs(rho),'phase_degrees':float(np.angle(rho,deg=True))}}
def rng(s,stream):return np.random.default_rng(np.random.SeedSequence([s,stream]))
def channels(g,r,n):
 f=K.cn(r,(n,1,2));b=g['alpha']*f+g['qr'][0,0]*K.cn(r,(n,1,2));return f,b
M.channels=channels
class World(K.World):
 def __init__(self,s,g):
  r=rng(s,1);n=8000
  self.inputs=r.choice([-1.,1.],(n,3));self.labels=(self.inputs[:,0]*self.inputs[:,1]>0).astype(float)
  self._f,self._b=channels(g,r,n);self._nf=.1*K.cn(r,(n,1));self._nb=.03*K.cn(r,(n,2));self._np=.1*K.cn(r,(n,1,2))
  # Independent streams per 64-pair block ensure all k share exactly nested observations.
  ff=[];bb=[]
  for block in range(16):
   cr=rng(s,200+block);f,b=channels(g,cr,64);ff.append(f+.1*K.cn(cr,f.shape));bb.append(b+.1*K.cn(cr,b.shape))
  self.cal_fhat=np.concatenate(ff);self.cal_bhat=np.concatenate(bb)
  tr=rng(s,3);n=8192
  self.test_inputs=tr.choice([-1.,1.],(n,3));self.test_labels=(self.test_inputs[:,0]*self.test_inputs[:,1]>0).astype(float)
  self._test_f=K.cn(tr,(n,1,2));self._test_n=.1*K.cn(tr,(n,1));self.test_epsilon=K.cn(tr,(n,2))
def cal(s,g,k,task,w=None):
 if task=='geometry':f=w.cal_fhat[:k];b=w.cal_bhat[:k];sigma=.1;quant=K.qcomplex
 else:
  ff=[];bb=[]
  for block in range(16):
   rr=rng(s,400+block);f,b=channels(g,rr,64);ff.append(f+.5*K.cn(rr,f.shape));bb.append(b+.03*K.cn(rr,b.shape))
  f=np.concatenate(ff)[:k];b=np.concatenate(bb)[:k];sigma=.5;quant=M.qc
 f,clips=quant(f,4)
 cov=float(np.mean(abs(f)**2)-sigma**2+.02);den=max(cov,.02)
 return complex(np.mean(b*f.conj())/den),{'forward_quant_clips':clips,'covariance_floor':cov<.02}
def post(a,m):
 if m in ['raw','csi']:return 1+0j
 phi=np.angle(a)
 if m=='phase2':phi=np.round(phi/(np.pi/2))*(np.pi/2)
 if m=='phase1':phi=0 if np.cos(phi)>=0 else np.pi
 if m=='scalar':
  if abs(a)==0:raise FloatingPointError('zero scalar estimate')
  return 1/a.conjugate()
 return np.exp(1j*phi)
def ledger(task,m,k,d,n):
 physical=m!='csi';pairs=0 if m in ['csi','oracle'] else k;slots=1 if task=='geometry' else 4
 pilot_rf=3*pairs;forward=slots*n;reverse=slots*n if physical else 0;csi=2*n if not physical else 0
 gainbits=16*slots*n if physical else 0;digitalbits=64*slots*n if not physical else 0;calbits=64*pairs
 return dict(training_samples=n,updates=int(np.ceil(n/32)),pilot_pairs=pairs,pilot_rf=pilot_rf,forward_rf=forward,reverse_rf=reverse,csi_pilot_rf=csi,rf_total=pilot_rf+forward+reverse+csi,calibration_control_bits=calbits,norm_control_bits=gainbits,digital_gradient_bits=digitalbits,total_control_bits=calbits+gainbits+digitalbits,phase_code_bits=1 if m=='phase1' else 2 if m=='phase2' else None,phase_code_transmitted=False,calibration_used=m not in ['raw','csi','oracle'],privileged_oracle=m=='oracle',return_wait_lambda=d,wait_seconds_example=d*(299792458/3.5e9)/15)
def train(task,s,g,m,k=64):
 if task=='geometry':
  w=World(s,g);a,ci=(g['alpha'],{'oracle':True}) if m=='oracle' else cal(s,g,k,task,w)
  p=post(a,m);eff=1/p.conjugate()
  K.C.update(scalar_calibration_pairs=64)
  K.fit_temporal=lambda *args,**kw:(np.array([[a]]),np.eye(1),eff,ci)
  row=K.train(w,s,str(g['d']),'csi_rf_budget' if m=='csi' else 'scalar_est',g['a']);n=row['costs']['training_labels']
 else:
  a,ci=(g['alpha'],{'oracle':True}) if m=='oracle' else cal(s,g,k,task)
  p=post(a,m);z0=M.norm(M.cn(M.rng(s,101),(8,2,4)))
  row=M.train('csi_rf' if m=='csi' else 'scalar_est',g,s,z0,np.eye(1),np.eye(1),p);n=5440 if m=='csi' else 4032
  row['service']={'ce':row['final']['ce'],'accuracy':1-row['final']['bler']};row.pop('ledger')
 row.update(task=task,seed=s,method=m,pilot_variant=k if m not in ['oracle','csi'] else 0,geometry=g['info'],calibration_alpha=pack(a),post=pack(p),calibration=ci,costs=ledger(task,m,k,g['d'],n))
 return row
def boot(v):
 v=np.array(v);ids=np.random.default_rng(95159999).integers(0,len(v),(20000,len(v)))
 return {'mean':float(v.mean()),'ci95':np.quantile(v[ids].mean(1),[.025,.975]).tolist(),'per_seed':v.tolist()}
def freeze(name):
 files=list((R/name/'src').glob('*.py'))+list((R/'shared/src').glob('*.py'))+list((R/'shared/configs').glob('*.json'))+[R/'infra/jobs.py']
 dump(R/name/'freeze_manifest.json',{'time':time.time(),'hashes':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}})
def verify():
 # Both adapted receive derivatives and complex encoder chain finite differences.
 rr=rng(95159900,99);errors=[]
 for task in ['geometry','message']:
  if task=='geometry':
   inp=rr.choice([-1.,1.],(17,3));lab=(inp[:,0]*inp[:,1]>0).astype(float);h=K.cn(rr,(17,1,2));W,w,b=K.initial(95159900);noise=.1*K.cn(rr,(17,1))
   def loss(z):return K.task(np.einsum('bri,bi->br',h,K.mv(z,inp))+noise,lab,w,b)[0].mean()
   y=np.einsum('bri,bi->br',h,K.mv(W,inp))+noise;s=K.task(y,lab,w,b)[1];grad=K.pgrad(K.pb(h,s),inp);z=W
  else:
   z=M.norm(K.cn(rr,(8,2,4)));h=K.cn(rr,(8,1,2));lab=np.arange(8);w=rr.normal(size=(8,16));b=rr.normal(size=8);noise=.5*K.cn(rr,(8,1,4))
   def loss(z):
    y=np.einsum('bri,bit->brt',h,M.norm(z))+noise
    return M.decode(M.features(y),w,b,lab)[0].mean()
   y=np.einsum('bri,bit->brt',h,z)+noise;p=M.decode(M.features(y),w,b,lab)[1];seed=M.backward(y,w,p);grad=M.pull(z,np.einsum('bri,brt->bit',h.conj(),seed))/8
  for ix in np.ndindex(z.shape):
   for part in [1,1j]:
    dz=np.zeros_like(z);dz[ix]=part;eps=1e-6
    errors.append(abs((loss(z+eps*dz)-loss(z-eps*dz))/(2*eps)-np.vdot(grad,dz).real))
 assert max(errors)<1e-7,max(errors)
 assert abs(geo('isotropic',D1)['alpha']-j0(2*np.pi*D1))<1e-12
 return {'max_fd_error':float(max(errors)),'checks':len(errors),'scalar_channel_dimensions':[1,2],'gate_eval_seeds_disjoint':not set(SEEDS)&set(GATE)}
def gate():
 p=R/'gate';dump(p/'results/preflight.json',verify());rows=[]
 for task in ['geometry','message']:
  for s in GATE:
   for m in ['csi','raw']:
    row=train(task,s,geo('isotropic',0),m);rows.append(row);print(task,s,m,row['service'],flush=True)
 dump(p/'results/all_runs.json',rows)
 scores={t:float(np.mean([r['service']['accuracy'] for r in rows if r['task']==t and r['method']=='csi'])) for t in ['geometry','message']}
 tasks=['geometry','message'] if min(scores.values())>=.9 else ['geometry'] if scores['geometry']>=.9 else ['message']
 dump(p/'results/decision.json',{'csi_mean_accuracy':scores,'learnable':{t:s>=.9 for t,s in scores.items()},'primary_task':tasks[0],'evaluation_tasks':tasks,'frozen_at':time.time()})
if __name__=='__main__':gate()
