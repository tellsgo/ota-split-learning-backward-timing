import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.special import logsumexp
C={'nr':1,'messages':8,'forward_noise_variance':.1}
IU=np.triu_indices(4,1)
N={'scalar_est':4032,'matrix_est':4032,'matrix_oracle':4032,'csi_rf':5440}
# Unused legacy branches are retained in train for traceable source comparison.
def ledger(*args):return {}
def emat(a, b):
    return np.einsum('ik,kj->ij', a, b, optimize=False)

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

def features(y):
    gram = np.einsum('bri,brj->bij', y.conj(), y) / C['nr']
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
    return 2 * np.einsum('bri,bij->brj', y, A) / C['nr']

def labels(s, n):
    r = rng(s, 105)
    return np.concatenate([r.permutation(np.repeat(np.arange(C['messages']), 32 // C['messages'])) for _ in range(n // 32)])

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

def evaluate(z, W, b, f, noise, lab):
    y = np.einsum('bri,bit->brt', f, norm(z)[lab]) + noise
    (loss, _, pred) = decode(features(y), W, b, lab)
    return {'ce': float(loss.mean()), 'bler': float(np.mean(pred != lab)), 'errors': int(sum(pred != lab))}

def train(method, g, s, z0, co, oracle, scalar):
    n = N[method]
    lab = labels(s, n)
    rr = rng(s, 103)
    (f, b) = channels(g, rr, n)
    rn = rng(s, 104)
    noise = np.sqrt(C['forward_noise_variance']) * cn(rn, (n, C['nr'], 4))
    nr = 0.03 * cn(rn, (n, 2, 4))
    pilotnoise = np.sqrt(C['forward_noise_variance']) * cn(rn, (n, C['nr'], 2))
    z = digital() if method == 'fixed_digital' else z0.copy()
    W = np.zeros((C['messages'], 16))
    bias = np.zeros(C['messages'])
    stats = {'norm_clips': 0, 'gradient_clips': 0, 'loss_clips': 0, 'zero_seed_columns': 0, 'nonzero_return_columns': 0, 'tiny_projection_columns': 0, 'max_seed_energy': 0.0, 'sum_restored_norm_squared': 0.0, 'max_forward_vector_energy': 0.0}
    optinfo = {}
    if method.startswith('fixed'):
        y = np.einsum('bri,bit->brt', f, z[lab]) + noise
        (W, bias, optinfo) = fitrx(features(y), lab)
        updates = optinfo['iterations']
    else:
        az = Adam((C['messages'], 2, 4, 2))
        aw = Adam(W.shape)
        ab = Adam(bias.shape)
        sr = rng(s, 106)
        base = np.zeros(C['messages'])
        seen = np.zeros(C['messages'])
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
                for m in range(C['messages']):
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
                    cm = co if method == 'matrix_est' else oracle if method == 'matrix_oracle' else np.eye(C['nr'])
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
    tn = np.sqrt(C['forward_noise_variance']) * cn(rng(s, 109), (8192, C['nr'], 4))
    tl = np.tile(np.arange(C['messages']), 1024)
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
