import os
from pathlib import Path
R=Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['TMPDIR']=str(R/'tmp');os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,hashlib,gzip,urllib.request,traceback,math
import numpy as np
from scipy.special import j0
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1)
sys.path.insert(0,str(R/'vendor_s'))
import core as S
import communication as M
SEEDS=list(range(73171001,73171009));BSEEDS=list(range(73172001,73172009));GATE=[73170001,73170002]
NEAR=[.38,.40];REF=[.55,.70];D1=.6098;T=16;PHYSICAL=64512;CSI=114688
BOOT=np.random.default_rng(73179999).integers(0,8,(20000,8))
DEADLINE=float('inf')

def dump(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def log(event,**kw):
 row={'unix':time.time(),'event':event,**kw}
 with (R/'logs/events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(row,ensure_ascii=False),flush=True)
def checktime(start=False):
 if time.time()>DEADLINE-(300 if start else 30):raise TimeoutError('Six-hour deadline / closeout reserve reached')
def rng(s,k):return np.random.default_rng(np.random.SeedSequence([s,k]))
def ct(a):return torch.from_numpy(np.asarray(a,dtype=np.complex64))
def real(a):return torch.from_numpy(np.asarray(a,dtype=np.float32))
def pack(a):return {'real':float(np.real(a)),'imag':float(np.imag(a))}
def boot(v):
 v=np.asarray(v);assert len(v)==8
 return {'mean':float(v.mean()),'ci95':np.quantile(v[BOOT].mean(1),[.025,.975]).tolist(),'per_seed':v.tolist()}
def positive(b):return b['mean']>0 and b['ci95'][0]>0

def dataset():
 names=['train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte'];arr=[];prov=[]
 for name in names:
  p=R/'data'/(name+'.gz');url='https://ossci-datasets.s3.amazonaws.com/mnist/'+p.name
  if not p.exists():
   log('download_start',url=url);urllib.request.urlretrieve(url,p);log('download_done',file=p.name)
  b=gzip.decompress(p.read_bytes());off=16 if 'images' in name else 8;a=np.frombuffer(b,dtype=np.uint8,offset=off).copy()
  arr.append(a.reshape(-1,784) if off==16 else a)
  prov.append({'url':url,'file':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'shape':list(arr[-1].shape)})
 assert arr[0].shape==(60000,784) and arr[2].shape==(10000,784)
 dump(R/'data/provenance.json',prov)
 return [torch.from_numpy(a.astype(np.float32)/255) if i%2==0 else torch.from_numpy(a.astype(np.int64)) for i,a in enumerate(arr)]

class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(784,64);self.fc2=nn.Linear(64,4*T)
 def forward(self,x):
  z=torch.view_as_complex(self.fc2(F.relu(self.fc1(x))).reshape(-1,2,T,2).__getattribute__('contigu'+'ous')())
  return z/(z.abs().square().sum(1,keepdim=True).sqrt().clamp_min(1e-12))
class Receiver(nn.Module):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(T*T,32);self.fc2=nn.Linear(32,10)
  nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
 def forward(self,x):return self.fc2(F.relu(self.fc1(x)))
def feat(y):
 t=y.shape[-1];g=y.conj().unsqueeze(-1)*y.unsqueeze(-2);i,j=torch.triu_indices(t,t,1)
 return torch.cat([g.diagonal(dim1=-2,dim2=-1).real,g[:,i,j].real,g[:,i,j].imag],1)
def new_models(s):
 torch.manual_seed(s);return Encoder(),Receiver()
def qnorm(a):
 clipped=int(((a>0)&((a<2**-20)|(a>2**20))).sum())
 code=torch.round((a.clamp_min(2**-20).log2().clamp(-20,20)+20)/40*65534)
 return torch.where(a>0,2**(code/65534*40-20),torch.zeros_like(a)),clipped

def ledger(m,n,t,d,k=64):
 physical=m!='csi';pairs=0 if m in ['oracle','csi'] else k
 return {'training_blocks':n,'updates':n//32,'forward_rf':t*n,'reverse_rf':t*n if physical else 0,'csi_pilot_rf':0 if physical else 2*n,'statistical_pilot_pairs':pairs,'statistical_pilot_rf':3*pairs,'rf_total':t*n*(2 if physical else 1)+(0 if physical else 2*n)+3*pairs,'norm_bits':16*t*n if physical else 0,'digital_gradient_bits':0 if physical else 64*t*n,'calibration_bits':64*pairs,'common_schedule_bits':192,'phase_code_bits_local':1 if m=='phase1' else None,'wait_lambda_per_block':d if physical else None,'wait_seconds_3p5GHz_15mps':d*(299792458/3.5e9)/15 if physical else None,'privileged_exact_rho':m=='oracle'}

def evaluate(enc,rx,x,lab,s,noisevar,stream=107):
 rr=rng(s,stream);nr=rng(s,stream+2);losses=[];correct=[]
 with torch.no_grad():
  for k in range(0,len(x),256):
   inp=x[k:k+256];l=lab[k:k+256];n=len(inp)
   h=ct(M.cn(rr,(n,2)));noise=ct(np.sqrt(noisevar)*M.cn(nr,(n,T)))
   y=(h[:,:,None]*enc(inp)).sum(1)+noise;pred=rx(feat(y));losses.append(F.cross_entropy(pred,l,reduction='none').numpy());correct.append((pred.argmax(1)==l).numpy())
 losses=np.concatenate(losses);correct=np.concatenate(correct)
 return {'ce':float(losses.mean()),'accuracy':float(correct.mean())},losses,correct

def train_a(s,d,m,nv,data,tag,k=64):
 checktime(True);t0=time.time();enc,rx=new_models(s);eo=torch.optim.Adam(enc.parameters(),lr=.01);ro=torch.optim.Adam(rx.parameters(),lr=.03)
 n=CSI if m=='csi' else PHYSICAL
 S.configure({'forward_noise_variance':nv,'messages':8});g=S.geo('isotropic',d,nr=1)
 if m in ['csi','oracle']:a=g['alpha'];ci={'oracle':m=='oracle','calibration_not_used':True}
 else:_,a,ci=S.calibrate(s,g,k)
 post=S.post(a,m);rho=float(j0(2*np.pi*d));xtrain,ytrain,xtest,ytest=data
 labrng=rng(s,105);idx=np.concatenate([labrng.permutation(len(xtrain)) for _ in range(math.ceil(n/len(xtrain)))])[:n]
 hrng=rng(s,103);erng=rng(s,203);nrng=rng(s,104);brng=rng(s,204);prng=rng(s,304)
 checks={'norm_clips':0,'digital_clips':0,'max_forward_power':0.,'max_reverse_power':0.};trace=[]
 checkpoint_dir=R/'checkpoints'/tag;checkpoint_dir.mkdir(exist_ok=True)
 def save(step):torch.save({'seed':s,'step':step,'encoder':enc.state_dict(),'receiver':rx.state_dict(),'noisevar':nv,'T':T,'method':m},checkpoint_dir/('step_%05d.pt'%step))
 save(0)
 for step in range(n//32):
  if step%64==0:checktime()
  ix=idx[step*32:(step+1)*32];inp=xtrain[ix];labels=ytrain[ix]
  h=ct(M.cn(hrng,(32,2)));e=ct(M.cn(erng,(32,2)));noise=ct(np.sqrt(nv)*M.cn(nrng,(32,T)));nb=ct(.03*M.cn(brng,(32,2,T)));pn=ct(np.sqrt(nv)*M.cn(prng,(32,2)))
  eo.zero_grad(set_to_none=True);ro.zero_grad(set_to_none=True)
  code=enc(inp);y=((h[:,:,None]*code.detach()).sum(1)+noise).requires_grad_(True)
  pred=rx(feat(y));loss=F.cross_entropy(pred,labels,reduction='sum');loss.backward();seed=y.grad.detach()
  with torch.no_grad():
   if m=='csi':
    grad=(h+pn).conj()[:,:,None]*seed[:,None,:]
    parts=torch.view_as_real(grad);checks['digital_clips']+=int((parts.abs()>32).sum());grad=torch.view_as_complex((torch.round(parts.clamp(-32,32)/32*32767)/32767*32).__getattribute__('contigu'+'ous')())
   else:
    amp=seed.abs();gain,nclip=qnorm(amp);checks['norm_clips']+=nclip
    tx=seed/amp.clamp_min(1e-30);back=rho*h+math.sqrt(max(0,1-rho*rho))*e
    grad=(back.conj()[:,:,None]*tx[:,None,:]+nb)*gain[:,None,:]*post
    checks['max_reverse_power']=max(checks['max_reverse_power'],float(tx.abs().square().max()))
   checks['max_forward_power']=max(checks['max_forward_power'],float(code.abs().square().sum(1).max()))
  code.backward(grad/32);eo.step()
  with torch.no_grad():
   for name,p in rx.named_parameters():
    p.grad.div_(32)
    if name.endswith('weight'):p.grad.add_(p,alpha=.0002)
  ro.step()
  with torch.no_grad():
   for p in rx.parameters():p.clamp_(-5,5)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
  if step%126==0 or step==n//32-1:
   trace.append({'update':step+1,'ce':float(loss.detach()/32),'accuracy':float((pred.argmax(1)==labels).float().mean())})
  if step+1 in [n//64,n//32]:save(step+1)
 service,ce,correct=evaluate(enc,rx,xtest,ytest,s,nv)
 training,tce,tcorr=evaluate(enc,rx,xtrain[:8192],ytrain[:8192],s,nv,307)
 np.savez_compressed(R/'results'/(tag+'_evaluation.npz'),ce=ce,correct=correct,train_ce=tce,train_correct=tcorr)
 assert checks['max_forward_power']<1.00001 and checks['max_reverse_power']<1.00001
 return {'tag':tag,'task':'A','seed':s,'displacement':d,'method':m,'pilot_pairs':k if m not in ['csi','oracle'] else 0,'noise_variance':nv,'service':service,'train_evaluation':training,'rho':rho,'calibration_alpha':pack(a),'post':pack(post),'calibration':ci,'checks':checks,'trace':trace,'costs':ledger(m,n,T,d,k),'seconds':time.time()-t0,'checkpoint_dir':str(checkpoint_dir.relative_to(R))}

# Rician generator deliberately replaces only the S channel routine.
def rician_channels(g,r,n):
 K=g.get('K',0);d=g['d'];rho=float(j0(2*np.pi*d))
 f0=M.cn(r,(n,1,2));e=M.cn(r,(n,1,2))
 if K==0:return f0,rho*f0+np.sqrt(max(0,1-rho*rho))*e
 phase=np.exp(2j*np.pi*r.uniform(size=(n,1,1)));ports=np.exp(1j*np.pi*np.cos(np.pi/3)*np.arange(2))[None,None,:]
 los=phase*ports;a=np.sqrt(K/(K+1));b=np.sqrt(1/(K+1))
 return a*los+b*f0,a*los*np.exp(2j*np.pi*d*np.cos(np.pi/3))+b*(rho*f0+np.sqrt(max(0,1-rho*rho))*e)
def bgeo(K,d):
 g=S.geo('isotropic',d,nr=1);a=(K*np.exp(2j*np.pi*d*np.cos(np.pi/3))+j0(2*np.pi*d))/(K+1)
 g.update(K=K,alpha=complex(a),a=np.array([[a]]));g['info'].update(K=K,rho=pack(a),magnitude=float(abs(a)),phase_degrees=float(np.angle(a,deg=True)))
 return g

def train_b(s,K,d,m,tag):
 checktime(True);t0=time.time();S.channels=rician_channels;M.channels=rician_channels;S.configure({'messages':8,'forward_noise_variance':.1})
 row=S.train('message',s,bgeo(K,d),m,64);row.update(tag=tag,task='B',K=K,seconds=time.time()-t0)
 return row

class Runner:
 def __init__(self):
  self.rows=[];self.index={};self.path=R/'results/runs.jsonl'
  if self.path.exists():
   for line in self.path.read_text().splitlines():
    row=json.loads(line);self.rows.append(row);self.index[row['tag']]=row
 def run(self,tag,fn,*args):
  if tag in self.index:return self.index[tag]
  log('task_start',tag=tag)
  try:row=fn(*args,tag=tag)
  except Exception as e:
   log('task_failed',tag=tag,error=repr(e));(R/'logs'/(tag+'_failure.txt')).write_text(traceback.format_exc());raise
  with self.path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  self.rows.append(row);self.index[tag]=row;log('task_done',tag=tag,service=row['service'],seconds=row['seconds']);return row
 def a(self,s,d,m,nv,data,stage='main',k=64):
  tag=f'A_{stage}_s{s}_d{d:.4f}_{m}_k{k}_v{nv}'
  return self.run(tag,train_a,s,d,m,nv,data,k=k) if False else self._a(tag,s,d,m,nv,data,k)
 def _a(self,tag,s,d,m,nv,data,k):
  def call(*args,tag):return train_a(*args,tag=tag,k=k)
  return self.run(tag,call,s,d,m,nv,data)
 def b(self,s,K,d,m):return self.run(f'B_s{s}_K{K}_d{d:.4f}_{m}',train_b,s,K,d,m)

def unit_checks():
 torch.manual_seed(12345);enc,rx=new_models(12345)
 nn.init.normal_(rx.fc2.weight,std=.1)
 x=torch.randn(4,784);h=ct(M.cn(rng(12345,1),(4,2)));noise=ct(.1*M.cn(rng(12345,2),(4,T)));lab=torch.arange(4)
 code=enc(x);loss=F.cross_entropy(rx(feat((h[:,:,None]*code).sum(1)+noise)),lab)
 direct=torch.autograd.grad(loss,list(enc.parameters()))
 code=enc(x);y=((h[:,:,None]*code.detach()).sum(1)+noise).requires_grad_(True)
 seed=torch.autograd.grad(F.cross_entropy(rx(feat(y)),lab),y)[0]
 chain=torch.autograd.grad(code,list(enc.parameters()),grad_outputs=h.conj()[:,:,None]*seed[:,None,:])
 err=max(float((a-b).abs().max()) for a,b in zip(direct,chain));assert err<1e-6
 yy=M.cn(rng(12345,3),(8,1,4));W=rng(12345,4).normal(size=(8,16));bb=np.zeros(8);labs=np.arange(8);M.C.update(nr=1,messages=8)
 y=ct(yy[:,0]).requires_grad_(True);wf=real(W);loss=F.cross_entropy(F.linear(feat(y),wf),torch.arange(8),reduction='sum');dy=torch.autograd.grad(loss,y)[0].numpy()
 p=M.decode(M.features(yy),W,bb,labs)[1];expected=M.backward(yy,W,p)[:,0];e=float(np.max(abs(dy-expected)));assert e<1e-4
 amp=torch.logspace(-7,7,200);q,c=qnorm(amp);qn,cn=M.qnorm(amp.numpy());qe=float(np.max(abs(q.numpy()-qn)/np.maximum(qn,1e-30)));assert qe<4e-6
 assert len(set(SEEDS+GATE+BSEEDS))==18
 out={'encoder_parameters':sum(p.numel() for p in enc.parameters()),'receiver_parameters':sum(p.numel() for p in rx.parameters()),'parameter_chain_max_abs_error':err,'S_backward_max_abs_error':e,'qnorm_max_relative_error':qe,'S_codebook_real_parameters':128,'physical_RF_phase64':ledger('phase',PHYSICAL,T,.55)['rf_total'],'CSI_RF':ledger('csi',CSI,T,0)['rf_total'],'seed_disjoint':True,'torch_version':torch.__version__,'numpy_version':np.__version__,'cpu_threads':torch.get_num_threads()}
 dump(R/'results/preflight.json',out);return out

def main_a():
 unit_checks();data=dataset();run=Runner();gatefile=R/'results/GATE_DECISION.json'
 if gatefile.exists():gate=json.loads(gatefile.read_text())
 else:
  levels=[];passed=False
  for nv in [.1,.025]:
   rows=[run.a(s,0.,'csi',nv,data,stage='gate') for s in GATE]
   passed=all(r['service']['accuracy']>=.85 and r['train_evaluation']['accuracy']>=.85 for r in rows)
   levels.append({'noise_variance':nv,'passed':passed,'rows':[{'tag':r['tag'],'test':r['service'],'training':r['train_evaluation']} for r in rows]})
   log('gate_level',noise_variance=nv,passed=passed)
   if passed:break
  gate={'passed':passed,'selected_noise_variance':nv,'levels':levels,'frozen_unix':time.time()};dump(gatefile,gate)
 if not gate['passed']:
  dump(R/'results/A_STOP.json',{'reason':'Task not learnable: both precheck seeds did not all reach 85% training and test accuracy under the fixed budget and single escalation','A1':'Not tested','A2':'Not tested','A3':'Not tested'});return
 nv=gate['selected_noise_variance']
 for s in SEEDS:run.a(s,0.,'csi',nv,data)
 ce=lambda s,d,m,k=64:run.a(s,d,m,nv,data,k=k)['service']['ce']
 for s in SEEDS:
  for d in REF+[1.05,1.20]:ce(s,d,'raw')
 b=boot([np.mean([ce(s,d,'raw') for d in REF])-np.mean([ce(s,d,'raw') for d in [1.05,1.20]]) for s in SEEDS]);a1={'raw_negative_minus_positive':b,'verdict':'Supported under this setting' if positive(b) else 'Falsification threshold triggered'};dump(R/'results/A1.json',a1);log('claim_done',claim='A1',result=a1)
 for s in SEEDS:
  for d in NEAR+REF:ce(s,d,'oracle')
 gap=np.array([np.mean([ce(s,d,'oracle') for d in NEAR])-np.mean([ce(s,d,'oracle') for d in REF]) for s in SEEDS]);gb=boot(gap)
 a2={'oracle_gap':gb}
 if not positive(gb):a2.update(verdict='Falsification threshold triggered',stopped='Oracle gap nonpositive or interval contains zero; phase1024-specific branch not run')
 else:
  for s in SEEDS:
   for d in NEAR:
    for k in [64,1024]:ce(s,d,'phase',k)
  gain=np.array([np.mean([ce(s,d,'phase',64)-ce(s,d,'phase',1024) for d in NEAR]) for s in SEEDS]);bg=gap[BOOT].mean(1);rat=gain[BOOT].mean(1)/bg;ci=np.quantile(rat,[.025,.975]).tolist();stable=bool(np.all(bg>0))
  verdict='Falsification threshold triggered' if stable and ci[0]>.5 else 'Supported under this setting' if stable and ci[1]<.5 else 'Not supported; falsification threshold not triggered'
  a2.update(pilot_gain=boot(gain),fraction={'mean':float(gain.mean()/gap.mean()),'ci95':ci,'nonpositive_bootstrap_denominators':int(sum(bg<=0))},verdict=verdict)
 dump(R/'results/A2.json',a2);log('claim_done',claim='A2',result=a2)
 for s in SEEDS:
  for d in NEAR:
   for m in ['phase','phase1','raw']:ce(s,d,m)
  ce(s,D1,'phase')
 methods=['phase','phase1','raw'];values=np.array([[[ce(s,d,m) for m in methods] for d in NEAR] for s in SEEDS]);wait=np.array([ce(s,D1,'phase') for s in SEEDS]);best=values.mean(0).argmin(1);diff=values[:,np.arange(2),best].mean(1)-wait
 resampled=values[BOOT].mean(1).min(2).mean(1)-wait[BOOT].mean(1);a3={'best_per_d':dict(zip(map(str,NEAR),[methods[i] for i in best])),'gain':{'mean':float(diff.mean()),'ci95':np.quantile(resampled,[.025,.975]).tolist(),'per_seed':diff.tolist()},'fixed_selection_sensitivity':boot(diff),'per_d':{str(d):boot(values[:,i,best[i]]-wait) for i,d in enumerate(NEAR)},'selection_recomputed_in_bootstrap':True};a3['verdict']='Supported under this setting' if positive(a3['gain']) else 'Falsification threshold triggered';dump(R/'results/A3.json',a3);log('claim_done',claim='A3',result=a3)

def main_b():
 run=Runner();grid=NEAR+REF+[D1];bounds=[]
 for K in [0,1,3,10]:
  for d in grid:
   a=bgeo(K,d)['alpha'];lb=(K-1)/(K+1) if K>1 else 0
   bounds.append({'K':K,'d':d,'rho':pack(a),'magnitude':abs(a),'lower_bound':lb,'margin':abs(a)-lb,'passes':bool(abs(a)>=lb-1e-12)})
 b1={'verdict':'Analytically valid; prespecified grid check passed' if all(x['passes'] for x in bounds) else 'Falsification threshold triggered','grid':bounds};dump(R/'results/B1.json',b1);log('claim_done',claim='B1',result=b1)
 gaps={}
 for K in [0,3,1,10]:
  for s in BSEEDS:
   for d in NEAR+REF:run.b(s,K,d,'oracle')
  v=[np.mean([run.b(s,K,d,'oracle')['service']['ce'] for d in NEAR])-np.mean([run.b(s,K,d,'oracle')['service']['ce'] for d in REF]) for s in BSEEDS];gaps[str(K)]=boot(v)
  dump(R/'results/B2_partial.json',gaps)
  if K==3 and gaps['3']['mean']>gaps['0']['mean']:
   b2={'gaps':gaps,'verdict':'Falsification threshold triggered','stopped':'G3 > G0; K1/K10 oracle not run'};break
 else:
  order=[0,1,3,10];means=[gaps[str(k)]['mean'] for k in order];mono=all(a>=b for a,b in zip(means,means[1:]));b2={'gaps':gaps,'monotone_means':mono,'adjacent_differences':{f'{a}_minus_{b}':boot(np.array(gaps[str(a)]['per_seed'])-gaps[str(b)]['per_seed']) for a,b in zip(order,order[1:])},'verdict':'Falsification threshold triggered' if means[-1]>=.05 else 'Supported under this setting' if mono and means[-1]<.02 else 'Not supported; falsification threshold not triggered'}
 dump(R/'results/B2.json',b2);log('claim_done',claim='B2',result=b2)
 ds=[d for d in grid if bgeo(10,d)['alpha'].real<0];assert sorted(ds)==sorted([.55,.70,D1])
 if ds:
  for s in BSEEDS:
   for d in ds:
    for m in ['raw','phase']:run.b(s,10,d,m)
  v=[np.mean([run.b(s,10,d,'raw')['service']['ce']-run.b(s,10,d,'phase')['service']['ce'] for d in ds]) for s in BSEEDS];b=boot(v);b3={'displacements':ds,'raw_minus_phase64':b,'per_d':{str(d):boot([run.b(s,10,d,'raw')['service']['ce']-run.b(s,10,d,'phase')['service']['ce'] for s in BSEEDS]) for d in ds},'verdict':'Supported under this setting' if positive(b) else 'Not supported (prespecified zero-effect threshold)'}
 else:b3={'verdict':'No applicable displacement in the grid; skipped'}
 dump(R/'results/B3.json',b3);log('claim_done',claim='B3',result=b3)

if __name__=='__main__':
 dump(R/'logs/process.json',{'pid':os.getpid(),'start':time.time(),'argv':sys.argv})
 try:
  if sys.argv[1]=='A':main_a()
  elif sys.argv[1]=='B':main_b()
  elif sys.argv[1]=='check':unit_checks()
  log('stage_complete',stage=sys.argv[1])
 except BaseException as e:
  log('stage_failure',stage=sys.argv[1],error=repr(e));(R/'logs'/('stage_'+sys.argv[1]+'_failure_'+str(int(time.time()))+'.txt')).write_text(traceback.format_exc());raise
