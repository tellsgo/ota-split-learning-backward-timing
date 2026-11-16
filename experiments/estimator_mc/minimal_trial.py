"""CPU-only audit. No training, no imports of historical modules, no source writes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json, time, resource, hashlib, importlib.util, sys, gzip
from pathlib import Path
import numpy as np
from scipy.special import j0, j1, jn_zeros, iv
from scipy.optimize import brentq

resource.setrlimit(resource.RLIMIT_CPU, (1200, 1200))
start = time.perf_counter()
cpu = time.process_time()
OUT = Path(__file__).resolve().parent / 'results'
OLD = Path(__file__).resolve().parent / 'inputs'
rng = np.random.default_rng(202610190317)
eta, L, G, delta2 = .01, 10., 1., 1.2
c = L*eta/2
d0 = float(jn_zeros(0,1)[0]/(2*np.pi))
d1 = float(jn_zeros(1,1)[0]/(2*np.pi))
slope = float(2*np.pi*j1(2*np.pi*d0))

def threshold(v,n=0):
    return 2*c*(v+n)/(1+np.sqrt(1-4*c*c*(1-v)*(v+n)))

widths=[]
for nt in (1,2):
    for noise in (0.,.0009):
        v,n = nt*delta2,nt*noise
        a = float(threshold(v,n))
        left=brentq(lambda d:j0(2*np.pi*d)-a,d0-.15,d0)
        right=brentq(lambda d:-j0(2*np.pi*d)-a,d0,d1)
        widths.append(dict(nt=nt,noise=noise,critical=a,leading_halfwidth=c*(v+n)/slope,
                           left=left,right=right,left_width=d0-left,right_width=right-d0))

# Isotropic complex quadratic: F(theta)=L/2 ||theta||^2, ||g||=1.
# Each complex port innovation has variance |delta|^2 (not |delta|^2/Nt).
mc=[]
N=400000
for nt in (1,2):
    z=(rng.normal(size=(N,nt))+1j*rng.normal(size=(N,nt)))/np.sqrt(2)
    for d in (.365,.38,d0,.4,.401,.42,d1):
        a=abs(float(j0(2*np.pi*d)))
        noisevar=(1-a*a)*delta2+.0009
        gh=np.sqrt(noisevar)*z.copy(); gh[:,0]+=a
        change=-eta*gh[:,0].real+L*eta**2/2*np.sum(abs(gh)**2,axis=1)
        exact=-eta*a+L*eta**2/2*(a*a+nt*noisevar)
        se=float(change.std(ddof=1)/np.sqrt(N))
        mc.append(dict(nt=nt,d=d,a=a,exact=exact,mc=float(change.mean()),se=se,
                       zscore=float((change.mean()-exact)/se)))

# Tight L=10, but curvature along gradient is 1: positive upper bound does not imply increase.
a=.04
counter={'objective':'F(x,y)=x^2/2+5*y^2; theta=(1,0); proper complex channel noise',
         'a':a,'delta2':delta2,
         'actual_change':-eta*a+eta**2/2*(a*a+5.5*(1-a*a)*delta2),
         'L_smooth_upper_bound':-eta*a+L*eta**2/2*(a*a+(1-a*a)*delta2)}
assert counter['actual_change']<0<counter['L_smooth_upper_bound']

# Fixed mean gain; shrink stepsize. No inverse is applied. Standard SGD gives identical prediction.
eta_scan=[]
a=abs(float(j0(2*np.pi*.38)));v=2*delta2
for step in (.0005,.001,.005,.01,.02):
    second=a*a+(1-a*a)*v+2*.0009
    eta_scan.append(dict(eta=step,expected_change=-step*a+L*step*step/2*second))

# von Mises exact ACF and quadrature; beta is motion minus spectrum centre.
angles=np.arange(32768)*2*np.pi/32768
vm=[]
for beta_deg in (0,35,60,90):
    beta=np.deg2rad(beta_deg)
    for kappa in (0,.7,.8,1,2,5,10,20):
        p=np.exp(kappa*(np.cos(angles)-1));p/=p.sum()
        rho=complex(np.sum(p*np.exp(2j*np.pi*d1*np.cos(angles-beta))))
        closed=complex(iv(0,np.sqrt(complex(kappa*kappa-(2*np.pi*d1)**2,
                                  2*kappa*2*np.pi*d1*np.cos(beta))))/iv(0,kappa))
        vm.append(dict(beta_deg=beta_deg,kappa=kappa,real=rho.real,imag=rho.imag,
                       magnitude=abs(rho),phase_deg=float(np.angle(rho,deg=True)),
                       fixed_pi_projection=-rho.real,onebit_projection=abs(rho.real),
                       integral_closed_error=abs(rho-closed)))

src=Path(__file__).resolve().parent/'vendor_s/reference_geo.py'
before=hashlib.sha256(src.read_bytes()).hexdigest()
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('audit_reference_geo',src)
geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
gg=geo.geo('isotropic',.38)
geometry={'d':.38,'siso_rho':float(j0(2*np.pi*.38)),
          'array_trace_rho':float(gg['alpha'].real),
          'array_A_singular_values':np.linalg.svd(gg['a'],compute_uv=False).tolist(),
          'array_A_singular_values_d0383':np.linalg.svd(geo.geo('isotropic',.383)['a'],compute_uv=False).tolist(),
          'source':'experiments/estimator_mc/vendor_s/reference_geo.py','source_sha256':before}
assert hashlib.sha256(src.read_bytes()).hexdigest()==before

# Pilot phase experiment with existing calibration formula: Nt=2, independent pilot blocks,
# forward variance .1, reverse variance .0009, 16-bit forward quantization, scalar ridge .02.
# Only E cos phase error is measured. No inference to CE/accuracy is made.
pilot=[]
R=3000
for d in (.38,.4,d1):
    rho=float(j0(2*np.pi*d));acc={64:[],256:[],1024:[]}
    for offset in range(0,R,100):
        shape=(min(100,R-offset),1024,2)
        def cn():return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
        f=cn();b=rho*f+np.sqrt(1-rho*rho)*cn()
        ff=f+np.sqrt(.1)*cn();bb=b+.03*cn()
        ff=(np.rint(np.clip(ff.real,-4,4)/4*32767)+1j*np.rint(np.clip(ff.imag,-4,4)/4*32767))*4/32767
        for k in acc:
            cov=np.mean(abs(ff[:,:k])**2,axis=(1,2))-.1+.02
            ah=np.mean(bb[:,:k]*ff[:,:k].conj(),axis=(1,2))/np.maximum(cov,.02)
            acc[k].extend((np.cos(np.angle(ah)-np.angle(complex(rho)))).tolist())
    for k,vals in acc.items():
        x=np.array(vals)
        pilot.append(dict(d=d,pairs=k,replicates=R,mean_cos_error=float(x.mean()),
                          se=float(x.std(ddof=1)/np.sqrt(R)),
                          effective_mean_gain=float(abs(rho)*x.mean()),
                          oracle_gain=abs(rho)))

# Reaggregate historical raw files, not hand-entered report numbers.
siso=OLD/'siso2'
with gzip.open(siso/'E2/results/all_runs.json.gz', 'rt') as input_file:
    rows=json.load(input_file)
hist=[]
for d in (.38,.4,.55,.7):
    for method,k in [('phase',64),('phase',1024),('oracle',0),('csi',0)]:
        rr=[r for r in rows if abs(r['geometry']['displacement_lambda']-d)<1e-10
            and r['method']==method and r['pilot_variant']==k]
        hist.append(dict(d=d,method=method,k=k,n=len(rr),
                         ce=float(np.mean([r['service']['ce'] for r in rr])),
                         accuracy=float(np.mean([r['service']['accuracy'] for r in rr]))))
def avg(ds,m,k):return np.mean([r['ce'] for r in hist if r['d'] in ds and r['method']==m and r['k']==k])
gap=avg([.38,.4],'oracle',0)-avg([.55,.7],'oracle',0)
improve=avg([.38,.4],'phase',64)-avg([.38,.4],'phase',1024)
verdicts={
    '\u5728\u672c\u8bbe\u5b9a\u4e0b\u652f\u6301':'Supported under this setting',
    '\u89e6\u53d1\u8bc1\u4f2a\u7ebf':'Falsification threshold reached',
    '\u672a\u83b7\u652f\u6301\uff0c\u672a\u89e6\u53d1\u660e\u786e\u8bc1\u4f2a\u7ebf':'Not supported; no clear falsification threshold reached',
}
def translate_verdicts(value):
    if isinstance(value, dict):
        return {key:translate_verdicts(item) for key,item in value.items()}
    if isinstance(value, list):
        return [translate_verdicts(item) for item in value]
    return verdicts.get(value,value) if isinstance(value,str) else value

history={'E2_cells':hist,'oracle_gap':gap,'pilot_improvement':improve,'fraction_gap_closed':improve/gap,
         'E2_primary':translate_verdicts(json.loads((siso/'E2/results/primary.json').read_text())),
         'G02_summary':translate_verdicts(json.loads((OLD/'g02/results/summary.json').read_text()))}

result=dict(seed=202610190317,N_mc=N,L=L,eta=eta,delta2_over_G=delta2,d0=d0,d1=d1,slope=slope,
            widths=widths,quadratic_mc=mc,counterexample=counter,eta_scan=eta_scan,
            von_mises=vm,geometry=geometry,pilot_phase=pilot,history=history,
            wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
(OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
assert max(x['integral_closed_error'] for x in vm)<1e-12
assert max(abs(x['zscore']) for x in mc)<6
print(json.dumps({k:result[k] for k in ('d0','d1','slope','widths','counterexample','geometry','eta_scan','pilot_phase','wall_seconds','cpu_seconds')},indent=2))
