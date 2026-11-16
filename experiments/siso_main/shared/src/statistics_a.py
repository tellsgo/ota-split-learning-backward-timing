from core import np,SEEDS,boot,geo,D1
def ce(rows, s, d, m, k=None, sp='isotropic'):
    a = [r['service']['ce'] for r in rows if r['seed'] == s and r['geometry']['displacement_lambda'] == d and (r['method'] == m) and (r['geometry']['spectrum'] == sp) and (k is None or r['pilot_variant'] == k)]
    assert len(a) == 1, (s, d, m, k, sp, len(a))
    return a[0]

def summary(e, rows):
    if e == 'E1':
        neg = [0.4, 0.45, 0.55, 0.7, 0.85]
        pos = [0.9, 1.05, 1.2, 1.35]
        v = [np.mean([ce(rows, s, d, 'raw') for d in neg]) - np.mean([ce(rows, s, d, 'raw') for d in pos]) for s in SEEDS]
        b = boot(v)
        return {'negative_lobe': neg, 'second_positive_lobe': pos, 'raw_CE_improvement': b, 'verdict': 'Supported under this setting' if b['mean'] >= 0.02 and b['ci95'][0] > 0 else 'Falsification threshold triggered', 'raw_minus_csi_negative': boot([np.mean([ce(rows, s, d, 'raw') - ce(rows, s, d, 'csi') for d in neg]) for s in SEEDS]), 'raw_minus_csi_positive': boot([np.mean([ce(rows, s, d, 'raw') - ce(rows, s, d, 'csi') for d in pos]) for s in SEEDS])}
    if e == 'E2':
        near = [0.38, 0.4]
        far = [0.55, 0.7]
        gap = np.array([np.mean([ce(rows, s, d, 'oracle') for d in near]) - np.mean([ce(rows, s, d, 'oracle') for d in far]) for s in SEEDS])
        gain = np.array([np.mean([ce(rows, s, d, 'phase', 64) - ce(rows, s, d, 'phase', 1024) for d in near]) for s in SEEDS])
        ids = np.random.default_rng(96169999).integers(0, 8, (20000, 8))
        bg = gap[ids].mean(1)
        bi = gain[ids].mean(1)
        ratio = bi / bg
        gr = boot(gap)
        rat = {'mean': float(gain.mean() / gap.mean()), 'ci95': np.quantile(ratio, [0.025, 0.975]).tolist(), 'nonpositive_bootstrap_denominator': int(sum(bg <= 0))}
        false = gr['mean'] < 0.02 or rat['mean'] >= 0.5
        support = gr['ci95'][0] > 0.02 and rat['ci95'][1] < 0.5 and (min(bg) > 0)
        return {'oracle_near_minus_reference': gr, 'pilot_improvement_64_to_1024': boot(gain), 'fraction_gap_closed': rat, 'verdict': 'Falsification threshold triggered' if false else 'Supported under this setting' if support else 'Insufficient evidence', 'stronger_than_array_0p25': gr['mean'] > 0.25, 'cross_task_comparison_caveat': 'array benchmark uses binary geometry CE; present message CE is 8-class. Numeric greater-than is descriptive, not controlled causal evidence.'}
    if e == 'E3':
        near = [0.38, 0.4, 0.42]
        methods = ['phase', 'phase2', 'phase1', 'scalar']
        best = {str(d): min(methods, key=lambda m: np.mean([ce(rows, s, d, m) for s in SEEDS])) for d in near}
        v = [np.mean([ce(rows, s, d, best[str(d)]) - ce(rows, s, D1, 'phase') for d in near]) for s in SEEDS]
        b = boot(v)
        return {'best_method_per_d': best, 'delay_improvement': b, 'verdict': 'Supported under this setting' if b['mean'] >= 0.02 and b['ci95'][0] > 0 else 'Falsification threshold triggered', 'selection_uncertainty_included': False, 'per_d': {str(d): boot([ce(rows, s, d, best[str(d)]) - ce(rows, s, D1, 'phase') for s in SEEDS]) for d in near}, 'd1': D1}
    if e == 'E4':
        iso = [0.55, 0.7, 1.05, 1.2]
        br = [0.1, 0.2, 0.3, 0.4, 0.5]

        def delta(sp, ds, m):
            return boot([np.mean([ce(rows, s, d, m, sp=sp) - ce(rows, s, d, 'phase', sp=sp) for d in ds]) for s in SEEDS])
        a = delta('isotropic', iso, 'phase1')
        b = delta('broad', br, 'phase2')
        near90 = [d for d in br if 60 <= abs(geo('broad', d)['info']['phase_degrees']) <= 120]
        c = delta('broad', near90, 'phase1') if near90 else None
        return {'isotropic_1bit_minus_continuous': a, 'broad_2bit_minus_continuous': b, 'near90_displacements': near90, 'broad_near90_1bit_minus_continuous': c, 'isotropic_noninferior': a['ci95'][1] < 0.02, 'broad_2bit_noninferior': b['ci95'][1] < 0.02, 'broad_1bit_degraded': c is not None and c['mean'] >= 0.02 and (c['ci95'][0] > 0), 'verdict': 'Supported under this setting' if a['ci95'][1] < 0.02 and b['ci95'][1] < 0.02 and c and (c['mean'] >= 0.02) and (c['ci95'][0] > 0) else 'Not all criteria met (see individual criteria)', 'per_d': {sp: {str(d): {m: delta(sp, [d], m) for m in ['phase1', 'phase2', 'scalar']} for d in ds} for (sp, ds) in [('isotropic', iso), ('broad', br)]}}
