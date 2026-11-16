import numpy as np
def pack(z):return {"real":np.asarray(z).real.tolist(),"imag":np.asarray(z).imag.tolist()}
def geo(spectrum, d, direction='parallel'):
    th = np.arange(128) * 2 * np.pi / 128
    mu = np.deg2rad(35)
    if spectrum == 'isotropic':
        p = np.ones(128)
    elif spectrum == 'broad':
        p = 0.7 * np.exp(2 * np.cos(th - mu)) / np.i0(2) + 0.3
    else:
        k = float(spectrum[1:])
        p = np.exp(k * (np.cos(th - mu) - 1))
    p /= p.sum()
    u = np.cos(th) if direction == 'parallel' else np.sin(th)
    s0 = np.exp(-2j * np.pi * np.arange(3)[:, None] * 0.5 * np.cos(th)) * np.sqrt(p)
    st = s0 * np.exp(2j * np.pi * d * u)
    r = s0 @ s0.conj().T
    t = st @ s0.conj().T
    a = np.linalg.solve(r.T, t.T).T
    q = r - a @ r @ a.conj().T
    q = (q + q.conj().T) / 2

    def root(m):
        (e, v) = np.linalg.eigh(m)
        assert e.min() > -1e-09
        return v * np.sqrt(np.maximum(e, 0)) @ v.conj().T
    rt = root(r)
    qr = root(q)
    jr = np.block([[rt, np.zeros((3, 3))], [a @ rt, qr]])
    err = np.linalg.norm(jr @ jr.conj().T - np.block([[r, t.conj().T], [t, r]]))
    assert err < 1e-09
    rho = np.trace(t) / np.trace(r)
    dop = 2 * np.pi * d * sum(p * u)
    info = {'spectrum': spectrum, 'direction': direction, 'displacement_lambda': d, 'rho': pack(rho), 'magnitude': float(abs(rho)), 'phase_degrees': float(np.angle(rho, deg=True)), 'doppler_degrees': float(np.rad2deg(dop)), 'joint_factor_error': float(err), 'R0_condition': float(np.linalg.cond(r))}
    return {'r0': r, 'rt': t, 'a': a, 'q': q, 'alpha': rho, 'info': info, 'jr': jr, 'root': rt, 'qr': qr}
