import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import pathlib, json, time, resource, signal, hashlib, platform, traceback
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
ROOT = pathlib.Path(__file__).resolve().parents[1]
C = json.loads((ROOT / 'configs/communication.json').read_text())
IU = np.triu_indices(4, 1)
METHODS = ['raw', 'scalar_est', 'matrix_est', 'matrix_oracle', 'csi_rf', 'csi_samples', 'score', 'fixed_random', 'fixed_digital']
N = {'raw': 4096, 'scalar_est': 4032, 'matrix_est': 4032, 'matrix_oracle': 4032, 'csi_rf': 5440, 'csi_samples': 4096, 'score': 8192, 'fixed_random': 8192, 'fixed_digital': 8192}

def emat(a,b):
    return np.einsum('ik,kj->ij',a,b,optimize=False)

def rng(s, n):
    return np.random.default_rng(np.random.SeedSequence([s, n]))

def cn(r, shape):
    return (r.normal(size=shape) + 1j * r.normal(size=shape)) / np.sqrt(2)

def pack(z):
    return {'real': z.real.tolist(), 'imag': z.imag.tolist()}

def norm(z):
    a = np.sqrt(np.sum(abs(z) ** 2, axis=-2, keepdims=True))
    return z / np.maximum(a, 1e-12)

def pull(z, g):
    a = np.sqrt(np.sum(abs(z) ** 2, axis=-2, keepdims=True))
    x = z / np.maximum(a, 1e-12)
    return (g - x * np.sum((x.conj() * g).real, axis=-2, keepdims=True)) / np.maximum(a, 1e-12)

def qreal(x, bound):
    clips = int(np.count_nonzero(abs(x) > bound))
    return (np.rint(np.clip(x, -bound, bound) / bound * 32767) / 32767 * bound, clips)

def qc(x, b):
    (r, a) = qreal(x.real, b)
    (i, c) = qreal(x.imag, b)
    return (r + 1j * i, a + c)

def qnorm(a):
    nz = a > 0
    clips = int(np.count_nonzero(nz & ((a < 2 ** (-20)) | (a > 2 ** 20))))
    q = np.zeros_like(a)
    code = np.rint((np.clip(np.log2(np.maximum(a[nz], 2 ** (-20))), -20, 20) + 20) / 40 * 65534)
    q[nz] = 2 ** (code / 65534 * 40 - 20)
    return (q, clips)

def geo(d):
    th = np.arange(128) * 2 * np.pi / 128
    mu = np.deg2rad(35)
    from scipy.special import iv, j0
    w = 0.7 * np.exp(2 * np.cos(th - mu)) / iv(0, 2) + 0.3
    w /= w.sum()
    s = np.exp(-2j * np.pi * np.arange(3)[:, None] * 0.5 * np.cos(th)) * np.sqrt(w)
    st = s * np.exp(2j * np.pi * d * np.cos(th))
    r = np.einsum('ik,jk->ij', s, s.conj())
    t = np.einsum('ik,jk->ij', st, s.conj())
    a = np.linalg.solve(r.T, t.T).T
    q = r - emat(emat(a, r), a.conj().T)
    q = (q + q.conj().T) / 2

    def sqrtm(m):
        (e, v) = np.linalg.eigh(m)
        assert e.min() > -1e-10
        return emat(v * np.sqrt(np.maximum(e, 0)), v.conj().T)
    root = sqrtm(r)
    qr = sqrtm(q)
    assert np.linalg.norm(emat(root, root.conj().T) - r) < 1e-10
    assert np.linalg.norm(emat(qr, qr.conj().T) + emat(emat(a, r), a.conj().T) - r) < 1e-10
    assert np.max(abs(np.diag(r) - 1)) < 1e-12
    return {'r': r, 't': t, 'a': a, 'q': q, 'root': root, 'qr': qr}

def channels(g, r, n):
    f = np.einsum('ij,bjk->bik', g['root'], cn(r, (n, 3, 2)))
    b = np.einsum('ij,bjk->bik', g['a'], f) + np.einsum('ij,bjk->bik', g['qr'], cn(r, (n, 3, 2)))
    return (f, b)

def corr(a):
    (u, s, v) = np.linalg.svd(a)
    inv = np.zeros_like(s)
    keep = s > 0.1
    inv[keep] = 1 / s[keep]
    c = emat(u * inv, v)
    (c, clips) = qc(c, 32)
    return (c, {'singular_values': s.tolist(), 'rank': int(sum(keep)), 'clips': clips})

def calibration(g, s):
    rr = rng(s, 102)
    (f, b) = channels(g, rr, 64)
    (f, clips) = qc(f + 0.5 * cn(rr, f.shape), 4)
    b = b + 0.03 * cn(rr, b.shape)
    cov = np.einsum('bik,bjk->ij', f, f.conj()) / 128 - 0.25 * np.eye(3)
    cross = np.einsum('bik,bjk->ij', b, f.conj()) / 128
    cov = (cov + cov.conj().T) / 2 + 0.02 * np.eye(3)
    (e, v) = np.linalg.eigh(cov)
    ev = np.maximum(e, 0.02)
    ah = emat(cross, emat(v / ev, v.conj().T))
    alpha = np.trace(cross) / sum(ev)
    (co, ci) = corr(ah)
    (oracle, oi) = corr(g['a'])
    scalar = 0j if abs(alpha) < 1e-12 else 1 / (alpha / abs(alpha) * max(abs(alpha), 0.1)).conjugate()
    return (co, oracle, scalar, {'A_est': pack(ah), 'A_true': pack(g['a']), 'relative_error': float(np.linalg.norm(ah - g['a']) / np.linalg.norm(g['a'])), 'matrix': ci, 'oracle': oi, 'scalar': [float(scalar.real), float(scalar.imag)], 'forward_quant_clips': clips, 'cov_floor_count': int(sum(e < 0.02))})

def features(y):
    gram = np.einsum('bri,brj->bij', y.conj(), y) / 3
    return np.concatenate([np.diagonal(gram, axis1=1, axis2=2).real, gram[:, IU[0], IU[1]].real, gram[:, IU[0], IU[1]].imag], axis=1)

def decode(feat, W, b, labels):
    logits = emat(feat, W.T) + b
    lp = logits - logsumexp(logits, axis=1)[:, None]
    loss = -lp[np.arange(len(labels)), labels]
    p = np.exp(lp)
    p[np.arange(len(labels)), labels] -= 1
    return (loss, p, logits.argmax(axis=1))

def backward(y, W, p):
    df = emat(p, W)
    A = np.zeros((len(y), 4, 4), complex)
    A[:, np.arange(4), np.arange(4)] = df[:, :4]
    upper = (df[:, 4:10] + 1j * df[:, 10:16]) / 2
    A[:, IU[0], IU[1]] = upper
    A[:, IU[1], IU[0]] = upper.conj()
    return 2 * np.einsum('bri,bij->brj', y, A) / 3

def digital():
    z = np.zeros((8, 2, 4), complex)
    z[:, 0, 0] = 1
    z[:, 1, 1] = 1
    for m in range(8):
        bits = [1 - 2 * (m >> j & 1) for j in range(3)]
        z[m, 0, 2] = bits[0] / np.sqrt(2)
        z[m, 1, 2] = bits[1] / np.sqrt(2)
        z[m, 0, 3] = bits[2]
    return z

def labels(s, n):
    r = rng(s, 105)
    return np.concatenate([r.permutation(np.repeat(np.arange(8), 4)) for _ in range(n // 32)])

class Adam:

    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, x, g, lr):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        return x - lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-08)

def rview(z):
    return np.stack([z.real, z.imag], axis=-1)

def cview(z):
    return z[..., 0] + 1j * z[..., 1]

def fitrx(feat, lab):

    def fun(v):
        W = v[:128].reshape(8, 16)
        b = v[128:]
        (loss, p, _) = decode(feat, W, b, lab)
        return (loss.mean() + 0.0001 * np.sum(W * W), np.r_[(emat(p.T, feat) / len(lab) + 0.0002 * W).ravel(), p.mean(axis=0)])
    res = minimize(fun, np.zeros(136), jac=True, method='L-BFGS-B', bounds=[(-5, 5)] * 136, options={'maxiter': 150, 'gtol': 1e-06, 'ftol': 1e-12, 'maxls': 30})
    grad = fun(res.x)[1]
    pg = grad.copy()
    pg[(res.x <= -5 + 1e-10) & (grad > 0)] = 0
    pg[(res.x >= 5 - 1e-10) & (grad < 0)] = 0
    return (res.x[:128].reshape(8, 16), res.x[128:], {'success': bool(res.success), 'message': str(res.message), 'iterations': res.nit, 'projected_gradient_max': float(max(abs(pg)))})

def evaluate(z, W, b, f, noise, lab):
    y = np.einsum('bri,bit->brt', f, norm(z)[lab]) + noise
    (loss, _, pred) = decode(features(y), W, b, lab)
    return {'ce': float(loss.mean()), 'bler': float(np.mean(pred != lab)), 'errors': int(sum(pred != lab))}

def ledger(method, n, nonzero):
    cal = 320 if method in ['scalar_est', 'matrix_est'] else 0
    if method in ['raw', 'scalar_est', 'matrix_est', 'matrix_oracle']:
        rf = 8 * n + cal
        bits = 64 * n + (12288 if cal else 0) + (288 if method in ['matrix_est', 'matrix_oracle'] else 0)
        energy = 4 * n + nonzero + cal
    elif method.startswith('csi'):
        rf = 6 * n
        bits = 256 * n
        energy = rf
    elif method == 'score':
        rf = 4 * n
        bits = 16 * n
        energy = rf
    else:
        rf = 4 * n
        bits = 0
        energy = rf
    return {'training_blocks': n, 'rf_vectors': rf, 'unused_rf': 32768 - rf, 'control_bits': bits, 'common_schedule_setup_bits': 192, 'training_rf_energy': energy, 'deployment_vectors': 4, 'deployment_energy': 4, 'payload_bits': 3}

def train(method, g, s, z0, co, oracle, scalar):
    n = N[method]
    lab = labels(s, n)
    rr = rng(s, 103)
    (f, b) = channels(g, rr, n)
    rn = rng(s, 104)
    noise = 0.5 * cn(rn, (n, 3, 4))
    nr = 0.03 * cn(rn, (n, 2, 4))
    pilotnoise = 0.5 * cn(rn, (n, 3, 2))
    z = digital() if method == 'fixed_digital' else z0.copy()
    W = np.zeros((8, 16))
    bias = np.zeros(8)
    stats = {'norm_clips': 0, 'gradient_clips': 0, 'loss_clips': 0, 'zero_seed_columns': 0, 'nonzero_return_columns': 0, 'tiny_projection_columns': 0, 'max_seed_energy': 0.0, 'sum_restored_norm_squared': 0.0, 'max_forward_vector_energy': 0.0}
    optinfo = {}
    if method.startswith('fixed'):
        y = np.einsum('bri,bit->brt', f, z[lab]) + noise
        (W, bias, optinfo) = fitrx(features(y), lab)
        updates = optinfo['iterations']
    else:
        az = Adam((8, 2, 4, 2))
        aw = Adam(W.shape)
        ab = Adam(bias.shape)
        sr = rng(s, 106)
        base = np.zeros(8)
        seen = np.zeros(8)
        for k in range(0, n, 32):
            sl = slice(k, k + 32)
            ll = lab[sl]
            zz = z[ll]
            eps = None
            if method == 'score':
                eps = 0.1 * (sr.normal(size=zz.shape) + 1j * sr.normal(size=zz.shape))
                zz = zz + eps
            x = norm(zz)
            stats['max_forward_vector_energy'] = max(stats['max_forward_vector_energy'], float(np.max(np.sum(abs(x) ** 2, axis=1))))
            y = np.einsum('bri,bit->brt', f[sl], x) + noise[sl]
            feat = features(y)
            (loss, p, _) = decode(feat, W, bias, ll)
            seed = backward(y, W, p)
            gw = emat(p.T, feat) / 32 + 0.0002 * W
            gb = p.mean(axis=0)
            if method == 'score':
                stats['loss_clips'] += int(sum(loss > 32))
                ql = np.rint(np.clip(loss, 0, 32) / 32 * 65535) / 65535 * 32
                gz = (ql - base[ll])[:, None, None] * eps / 0.01
                for m in range(8):
                    take = ll == m
                    total = seen[m] + sum(take)
                    base[m] = (base[m] * seen[m] + sum(ql[take])) / total
                    seen[m] = total
            else:
                if method.startswith('csi'):
                    gx = np.einsum('bri,brt->bit', (f[sl] + pilotnoise[sl]).conj(), seed)
                    (gx, clips) = qc(gx, 32)
                    stats['gradient_clips'] += clips
                else:
                    cm = co if method == 'matrix_est' else oracle if method == 'matrix_oracle' else np.eye(3)
                    v = np.einsum('ij,bjt->bit', cm, seed)
                    a = np.sqrt(np.sum(abs(v) ** 2, axis=1))
                    tx = v / np.maximum(a[:, None, :], 1e-30)
                    (gain, clips) = qnorm(a)
                    stats['norm_clips'] += clips
                    stats['zero_seed_columns'] += int(sum((a == 0).ravel()))
                    stats['nonzero_return_columns'] += int(sum((a > 0).ravel()))
                    stats['max_seed_energy'] = max(stats['max_seed_energy'], float(np.max(np.sum(abs(tx) ** 2, axis=1))))
                    stats['sum_restored_norm_squared'] += float(np.sum(gain * gain))
                    gx = (np.einsum('bri,brt->bit', b[sl].conj(), tx) + nr[sl]) * gain[:, None, :]
                    if method == 'scalar_est':
                        gx *= scalar
                gz = pull(zz, gx)
            full = np.zeros_like(z)
            np.add.at(full, ll, gz / 32)
            zr = cview(az.step(rview(z), rview(full), 0.01))
            tiny = np.sum(abs(zr) ** 2, axis=1, keepdims=True) < 1e-24
            stats['tiny_projection_columns'] += int(sum(tiny.ravel()))
            z = np.where(tiny, z, norm(zr))
            W = np.clip(aw.step(W, gw, 0.03), -5, 5)
            bias = np.clip(ab.step(bias, gb, 0.03), -5, 5)
            assert np.all(np.isfinite(z)) and np.all(np.isfinite(W))
        updates = n // 32
    stats['max_forward_vector_energy'] = max(stats['max_forward_vector_energy'], float(np.max(np.sum(abs(z) ** 2, axis=1))))
    assert stats['max_forward_vector_energy'] <= 1 + 1e-10 and stats['max_seed_energy'] <= 1 + 1e-10
    (tf, _) = channels(g, rng(s, 107), 8192)
    tn = 0.5 * cn(rng(s, 109), (8192, 3, 4))
    tl = np.tile(np.arange(8), 1024)
    initial = evaluate(digital() if method == 'fixed_digital' else z0, np.zeros_like(W), np.zeros_like(bias), tf, tn, tl)
    out = {'seed': s, 'displacement': g['d'], 'method': method, 'initial': initial, 'final': evaluate(z, W, bias, tf, tn, tl), 'initial_codebook': pack(digital() if method == 'fixed_digital' else z0), 'final_codebook': pack(z), 'receiver_weights': W.tolist(), 'receiver_bias': bias.tolist(), 'checks': stats, 'optimizer': optinfo, 'updates': updates, 'ledger': ledger(method, n, stats['nonzero_return_columns'])}
    if method == 'fixed_digital':
        y = np.einsum('bri,bit->brt', tf, z[tl]) + tn
        fh = y[:, :, :2]
        means = np.einsum('bri,mit->bmrt', fh, z[:, :, 2:])
        dist = np.sum(abs(y[:, None, :, 2:] - means) ** 2, axis=(2, 3))
        pred = dist.argmin(axis=1)
        out['plugin_pilot_detector'] = {'bler': float(np.mean(pred != tl)), 'errors': int(sum(pred != tl)), 'training_blocks': 0, 'deployment_vectors': 4, 'deployment_energy': 4}
    return out

def verify():
    r = rng(5000999, 108)
    z = norm(cn(r, (5, 2, 4)))
    h = cn(r, (5, 3, 2))
    noise = 0.5 * cn(r, (5, 3, 4))
    W = r.normal(size=(8, 16)) * 0.2
    bias = r.normal(size=8) * 0.1
    lab = np.arange(5)

    def fun(zz, ww, bb):
        yy = np.einsum('bri,bit->brt', h, norm(zz)) + noise
        return decode(features(yy), ww, bb, lab)[0].mean()
    y = np.einsum('bri,bit->brt', h, norm(z)) + noise
    p = decode(features(y), W, bias, lab)[1]
    grad = pull(z, np.einsum('bri,brt->bit', h.conj(), backward(y, W, p))) / 5
    err = []
    for ix in np.ndindex(z.shape):
        for part in [1, 1j]:
            dz = np.zeros_like(z)
            dz[ix] = part
            fd = (fun(z + 1e-06 * dz, W, bias) - fun(z - 1e-06 * dz, W, bias)) / 2e-06
            err.append(abs(fd - np.vdot(grad, dz).real))
    assert max(err) < 1e-07, max(err)
    zc = norm(cn(r, (2, 4)))
    v = cn(r, (2, 4))
    eps = 0.1 * (r.normal(size=(80000, 2, 4)) + 1j * r.normal(size=(80000, 2, 4)))
    plus = zc + eps
    minus = zc - eps
    lp = np.sum((v.conj() * norm(plus)).real, axis=(1, 2))
    lm = np.sum((v.conj() * norm(minus)).real, axis=(1, 2))
    score = (lp - lm)[:, None, None] * eps / (2 * 0.01)
    path = (pull(plus, np.broadcast_to(v, plus.shape)) + pull(minus, np.broadcast_to(v, minus.shape))) / 2
    diff = rview(score - path)
    mean = diff.mean(axis=0)
    se = diff.std(axis=0, ddof=1) / np.sqrt(len(diff))
    zmax = float(np.max(abs(mean) / se))
    assert zmax < 6, zmax
    models = {}
    for d in C['displacements']:
        g = geo(d)
        (f, b) = channels(g, r, 12000)
        cov = np.einsum('bik,bjk->ij', f, f.conj()) / 24000
        bc = np.einsum('bik,bjk->ij', b, b.conj()) / 24000
        cross = np.einsum('bik,bjk->ij', b, f.conj()) / 24000
        errors = [float(np.linalg.norm(cov - g['r'])), float(np.linalg.norm(bc - g['r'])), float(np.linalg.norm(cross - g['t']))]
        assert max(errors) < 0.12, errors
        (cu, sv, vh) = np.linalg.svd(g['a'])
        exact = emat(cu * (1 / sv), vh)
        assert np.linalg.norm(emat(g['a'].conj().T, exact) - np.eye(3)) < 1e-10
        models[str(d)] = {'covariance_errors': errors, 'R0': pack(g['r']), 'Rtau': pack(g['t']), 'A': pack(g['a']), 'conditional_inverse_residual': float(np.linalg.norm(emat(g['a'].conj().T, exact) - np.eye(3)))}
    assert np.max(abs(features(y * np.exp(0.7j)) - features(y))) < 1e-12
    assert np.max(abs(backward(y, np.zeros_like(W), p))) == 0
    assert qnorm(np.zeros((2, 4)))[0].sum() == 0
    return {'finite_difference_max': float(max(err)), 'score_normalized_objective_moment_max_z': zmax, 'models': models, 'zero_seed': True, 'global_phase_gauge': True, 'normalization_convention': 'score directly in latent; physical/CSI local Jacobian'}

def summarize(rows):
    out = {}
    br = np.random.default_rng(C['bootstrap_seed'])
    for d in C['displacements']:
        sub = [x for x in rows if x['displacement'] == d]
        arms = {m: {'mean_ce': float(np.mean([x['final']['ce'] for x in sub if x['method'] == m])), 'mean_bler': float(np.mean([x['final']['bler'] for x in sub if x['method'] == m]))} for m in METHODS}
        gains = {}
        raw = np.array([next((x['final']['ce'] for x in sub if x['seed'] == s and x['method'] == 'raw')) for s in C['seeds']])
        for m in ['scalar_est', 'matrix_est']:
            vals = np.array([next((x['final']['ce'] for x in sub if x['seed'] == s and x['method'] == m)) for s in C['seeds']])
            delta = raw - vals
            boots = delta[br.integers(0, 8, (10000, 8))].mean(axis=1)
            ci = np.quantile(boots, [0.025, 0.975])
            gains[m] = {'mean_gain': float(delta.mean()), 'median_gain': float(np.median(delta)), 'ci95': ci.tolist(), 'wins': int(sum(delta > 0)), 'per_seed': delta.tolist(), 'gate': bool(delta.mean() >= 0.02 and ci[0] > 0 and (sum(delta > 0) >= 6))}
        out[str(d)] = {'arms': arms, 'raw_repair': gains, 'pilot_detector_mean_bler': float(np.mean([x['plugin_pilot_detector']['bler'] for x in sub if 'plugin_pilot_detector' in x]))}
    return out

def main():
    resource.setrlimit(resource.RLIMIT_CPU, (C['cpu_seconds'], C['cpu_seconds']))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('wall cap')))
    signal.alarm(C['wall_seconds'])
    manifest = json.loads((ROOT / 'results/freeze_manifest.json').read_text())
    for (p, sha) in manifest['hashes'].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha
    started = time.time()
    checks = verify()
    (ROOT / 'results/preflight.json').write_text(json.dumps(checks, indent=2))
    print('PREFLIGHT PASS', flush=True)
    rows = []
    cal = []
    for s in C['seeds']:
        z0 = norm(cn(rng(s, 101), (8, 2, 4)))
        for d in C['displacements']:
            g = geo(d)
            g['d'] = d
            (co, oracle, scalar, ci) = calibration(g, s)
            cal.append({'seed': s, 'displacement': d, **ci})
            for m in METHODS:
                row = train(m, g, s, z0, co, oracle, scalar)
                rows.append(row)
            (ROOT / 'results/partial_runs.json').write_text(json.dumps(rows))
            (ROOT / 'results/calibration.json').write_text(json.dumps(cal, indent=2))
            print(json.dumps({'seed': s, 'displacement': d, 'rows': len(rows), 'wall': time.time() - started}), flush=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {'metadata': {'host': platform.node(), 'pid': os.getpid(), 'task': C['task'], 'rows': len(rows), 'wall_seconds': time.time() - started, 'user_cpu_seconds': usage.ru_utime, 'system_cpu_seconds': usage.ru_stime}, 'cases': summarize(rows)}
    (ROOT / 'results/all_runs.json').write_text(json.dumps(rows, indent=2))
    (ROOT / 'results/summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (ROOT / 'results/failure.txt').write_text(traceback.format_exc())
        raise
