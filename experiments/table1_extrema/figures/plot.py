import os, pathlib, json, csv
R=pathlib.Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(R/'figures/cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
S=json.loads((R/'results/summary.json').read_text()); rows=json.loads((R/'results/all_runs.json').read_text())
with (R/'figures/data.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['d','seed','method','ce','accuracy','rho_real','rho_imag','phase_error_degrees'])
 for r in rows:
  a=r['calibration_alpha'];a=complex(a['real'],a['imag']);rho=r['geometry']['rho'];rho=complex(rho['real'],rho['imag'])
  e=float(np.angle(a/rho,deg=True)) if r['arm']!='CSI' else ''
  w.writerow([r['displacement'],r['seed'],r['arm'],r['service']['ce'],r['service']['accuracy'],rho.real,rho.imag,e])
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for arm in ['phase_64','uncorrected','CSI']:
 rr=[r for r in S['arms'] if r['arm']==arm];x=np.array([r['d'] for r in rr]);y=np.array([r['ce']['mean'] for r in rr]);ci=np.array([r['ce']['ci95'] for r in rr])
 ax[0].errorbar(x,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),marker='o',capsize=3,label=arm)
ax[0].set(xlabel='Displacement / wavelength',ylabel='Final test CE',title='Fixed budget; 8 independent seeds');ax[0].legend();ax[0].grid(alpha=.2)
labels=['d1 - CSI','d2 - CSI','d2 - d1','d3 - d1'];cl=S['claims'];ss=[cl['G2.1']['phase_minus_CSI'],cl['G2.2']['phase_minus_CSI'],cl['G2.2']['phase_minus_d1'],cl['G2.3']['phase_minus_d1']]
y=np.array([s['mean'] for s in ss]);ci=np.array([s['ci95'] for s in ss]);ax[1].errorbar(range(4),y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=4)
ax[1].set_xticks(range(4),labels,rotation=15);ax[1].axhline(.02,color='r',ls='--',label='0.02');ax[1].axhline(.04,color='orange',ls=':',label='0.04');ax[1].axhline(0,color='gray',lw=.8);ax[1].set(ylabel='Paired CE difference',title='Paired seed bootstrap 95% intervals');ax[1].legend();ax[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'figures/g02.png',dpi=180);fig.savefig(R/'figures/g02.pdf');plt.close(fig)
print('Figures and source data written')
