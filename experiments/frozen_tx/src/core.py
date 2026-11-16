import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import pathlib,json,time,hashlib
import numpy as np
from scipy.special import jn_zeros
import communication as M
from reference_geo import geo as array_geo
R=pathlib.Path(__file__).resolve().parents[2]
SEEDS=list(range(96161001,96161009));GATE=[96160001,96160002];D1=float(jn_zeros(1,1)[0]/(2*np.pi))
CONFIG={'messages':8,'forward_noise_variance':.1}
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def pack(z):return {'real':np.asarray(z).real.tolist(),'imag':np.asarray(z).imag.tolist()}
def rng(s,stream):return M.rng(s,stream)
def configure(c):CONFIG.update(c)
def geo(sp,d,direction='parallel',nr=1):
 if nr==3:
  g=array_geo(sp,d,direction);g['r']=g['r0'];g['t']=g['rt'];g['d']=d;g['nr']=nr;return g
 th=np.arange(128)*2*np.pi/128;mu=np.deg2rad(35)
 p=np.ones(128) if sp=='isotropic' else .7*np.exp(2*np.cos(th-mu))/np.i0(2)+.3
 p/=p.sum();u=np.cos(th) if direction=='parallel' else np.sin(th)
 rho=complex(sum(p*np.exp(2j*np.pi*d*u)))
 if d==0:rho=1+0j
 q=max(0.,1-abs(rho)**2);a=np.array([[rho]]);one=np.ones((1,1))
 return {'r':one,'r0':one,'t':a,'rt':a,'a':a,'q':np.array([[q]]),'root':one,'qr':np.array([[np.sqrt(q)]]),'alpha':rho,'d':d,'nr':1,'info':{'spectrum':sp,'direction':direction,'displacement_lambda':d,'rho':pack(rho),'magnitude':abs(rho),'phase_degrees':float(np.angle(rho,deg=True))}}
def channels(g,r,n):
 nr=g['nr'];f=np.einsum('ij,bjk->bik',g['root'],M.cn(r,(n,nr,2)))
 b=np.einsum('ij,bjk->bik',g['a'],f)+np.einsum('ij,bjk->bik',g['qr'],M.cn(r,(n,nr,2)))
 return f,b
M.channels=channels
def calibrate(s,g,k):
 ff=[];bb=[];sigma=np.sqrt(CONFIG['forward_noise_variance'])
 for block in range((k+63)//64):
  rr=rng(s,400+block);f,b=channels(g,rr,64);ff.append(f+sigma*M.cn(rr,f.shape));bb.append(b+.03*M.cn(rr,b.shape))
 f,clips=M.qc(np.concatenate(ff)[:k],4);b=np.concatenate(bb)[:k];nr=g['nr']
 cov=np.einsum('bik,bjk->ij',f,f.conj())/(2*k)-sigma**2*np.eye(nr)
 cov=(cov+cov.conj().T)/2+.02*np.eye(nr);e,v=np.linalg.eigh(cov);ev=np.maximum(e,.02)
 cross=np.einsum('bik,bjk->ij',b,f.conj())/(2*k);ah=cross@(v/ev)@v.conj().T
 alpha=complex(np.trace(cross)/sum(ev))
 return ah,alpha,{'forward_quant_clips':clips,'covariance_floor_count':int(sum(e<.02)),'A_est':pack(ah)}
def polar(a):
 u,s,vh=np.linalg.svd(a.conj().T);w=u@vh;return w.conj().T,s

def post(a,m):
 if m in ['raw','csi']:return 1+0j
 phi=np.angle(a)
 if m=='phase2':phi=np.round(phi/(np.pi/2))*(np.pi/2)
 if m=='phase1':phi=0 if np.cos(phi)>=0 else np.pi
 if m=='scalar':
  if abs(a)==0:raise FloatingPointError('zero scalar estimate')
  return 1/a.conjugate()
 return np.exp(1j*phi)
def ledger(m,k,d,n,nr):
 oracle=m in ['oracle','polar_oracle'];physical=m!='csi';pairs=0 if oracle or not physical else k
 matrix=m in ['polar64','polar_oracle'];forward=4*n;reverse=4*n if physical else 0;csi=2*n if not physical else 0
 pilot=(2+nr)*pairs;cal=32*nr*2*pairs;norm=64*n if physical else 0;digital=256*n if not physical else 0;mat=32*nr*nr if matrix else 0
 return dict(training_samples=n,updates=n//32,pilot_pairs=pairs,pilot_rf=pilot,forward_rf=forward,reverse_rf=reverse,csi_pilot_rf=csi,rf_total=pilot+forward+reverse+csi,calibration_control_bits=cal,norm_control_bits=norm,digital_gradient_bits=digital,matrix_control_bits=mat,total_control_bits=cal+norm+digital+mat,phase_code_bits=1 if m=='phase1' else 2 if m=='phase2' else None,phase_code_transmitted=False,calibration_used=m not in ['raw','csi','oracle','polar_oracle'],privileged_oracle=oracle,matrix_quantized_16bit=m=='polar64',exact_matrix_privileged_real_parameters=2*nr*nr if m=='polar_oracle' else 0,matrix_bits_are_nominal_proxy=m=='polar_oracle',common_schedule_setup_bits=192,receive_antennas=nr,rf_metric='RF vector uses, not scalar observations or antenna hardware',return_wait_lambda=d if physical else 0,wait_seconds_example=d*(299792458/3.5e9)/15 if physical else 0)
def train(task,s,g,m,k=64):
 assert task=='message';nr=g['nr'];M.C.update(CONFIG,nr=nr)
 if m in ['oracle','polar_oracle','csi']:
  ah,a,ci=g['a'],g['alpha'],{'oracle':m!='csi','calibration_not_used':True}
 else:ah,a,ci=calibrate(s,g,k)
 p=post(a,m);co=np.eye(nr,dtype=complex);oracle=co.copy();method='csi_rf' if m=='csi' else 'scalar_est'
 if m in ['polar64','polar_oracle']:
  cm,svals=polar(ah);ci['polar_singular_values']=svals.tolist()
  if m=='polar64':cm,ci['matrix_quant_clips']=M.qc(cm,1)
  co=oracle=cm;method='matrix_est' if m=='polar64' else 'matrix_oracle';p=1+0j
 z0=M.norm(M.cn(M.rng(s,101),(CONFIG['messages'],2,4)))
 row=M.train(method,g,s,z0,co,oracle,p);n=M.N[method];row.pop('ledger')
 row.update(task=task,nr=nr,config=CONFIG.copy(),seed=s,method=m,pilot_variant=k if m not in ['oracle','polar_oracle','csi'] else 0,geometry=g['info'],calibration_alpha=pack(a),post=pack(p),correction_matrix=pack(co),calibration=ci,service={'ce':row['final']['ce'],'accuracy':1-row['final']['bler']},costs=ledger(m,k,g['d'],n,nr))
 return row
def boot(v):
 v=np.array(v);ids=np.random.default_rng(96169999).integers(0,len(v),(20000,len(v)))
 return {'mean':float(v.mean()),'ci95':np.quantile(v[ids].mean(1),[.025,.975]).tolist(),'per_seed':v.tolist()}
def freeze(e):
 files=list((R/e/'src').glob('*.py'))+list((R/'shared/src').glob('*.py'))+list((R/'shared/configs').glob('*.json'))+[R/'infra/jobs.py']
 dump(R/e/'freeze_manifest.json',{'time':time.time(),'hashes':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}})
def verify_freeze(e):
 man=json.loads((R/e/'freeze_manifest.json').read_text())
 for f,h in man['hashes'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f

def verify():
 rr=rng(96169900,99);errs=[];polar_errors=[];geometry_errors=[]
 for nr in [1,3]:
  for messages in [4,8]:
   M.C.update(nr=nr,messages=messages)
   z=M.norm(M.cn(rr,(messages,2,4)));h=M.cn(rr,(messages,nr,2));lab=np.arange(messages);w=rr.normal(size=(messages,16));b=rr.normal(size=messages);noise=.2*M.cn(rr,(messages,nr,4))
   def loss(zz):return M.decode(M.features(np.einsum('bri,bit->brt',h,M.norm(zz))+noise),w,b,lab)[0].mean()
   y=np.einsum('bri,bit->brt',h,z)+noise;p=M.decode(M.features(y),w,b,lab)[1]
   grad=M.pull(z,np.einsum('bri,brt->bit',h.conj(),M.backward(y,w,p)))/messages
   for ix in np.ndindex(z.shape):
    for part in [1,1j]:
     dz=np.zeros_like(z);dz[ix]=part;eps=1e-6;errs.append(abs((loss(z+eps*dz)-loss(z-eps*dz))/(2*eps)-np.vdot(grad,dz).real))
  for sp in ['isotropic','broad']:
   for d in [.1,.3,.38,.4,.55,.7,1.2]:
    g=geo(sp,d,nr=nr);ref=array_geo(sp,d)
    geometry_errors.append(abs(g['alpha']-ref['alpha']))
    if nr==3:geometry_errors.append(np.linalg.norm(g['a']-ref['a']))
    cm,sv=polar(g['a']);p=g['a'].conj().T@cm
    polar_errors.extend([np.linalg.norm(p-p.conj().T),max(0.,-np.linalg.eigvalsh(p).min()),np.linalg.norm(cm.conj().T@cm-np.eye(nr))])
    if nr==1:polar_errors.append(abs(cm[0,0]-np.exp(1j*np.angle(g['alpha']))))
 assert max(errs)<1e-7,max(errs)
 assert max(polar_errors)<1e-10,max(polar_errors)
 assert max(geometry_errors)<1e-12,max(geometry_errors)
 assert not set(SEEDS)&set(GATE)
 return {'gradient_fd_checks':len(errs),'gradient_max_error':float(max(errs)),'polar_max_error':float(max(polar_errors)),'reference_geometry_max_error':float(max(geometry_errors)),'gate_eval_disjoint':True}
