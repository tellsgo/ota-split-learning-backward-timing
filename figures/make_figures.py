"""Figures for the single-antenna letter."""
import csv, json, pathlib, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42; matplotlib.rcParams["ps.fonttype"] = 42  # TrueType, no Type 3 fonts (register pass 2026-11-14)
import matplotlib.pyplot as plt
from scipy.special import j0, jn_zeros

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = HERE
for name in ("E1", "E2", "E3"):
    source = ROOT / "experiments" / "siso_main" / name / "results" / "all_runs.json"
    if not source.exists():
        raise FileNotFoundError(f"Run the {name} target before figures: {source}")
    rows = json.loads(source.read_text())
    with (RUN / f"{name}_data.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["seed", "nr", "spectrum", "d", "method", "pilot_variant", "ce", "accuracy", "rho_real", "rho_imag"])
        writer.writeheader()
        for row in rows:
            geometry = row["geometry"]
            writer.writerow(dict(seed=row["seed"], nr=row["nr"], spectrum=geometry["spectrum"], d=geometry["displacement_lambda"], method=row["method"], pilot_variant=row["pilot_variant"], ce=row["service"]["ce"], accuracy=row["service"]["accuracy"], rho_real=geometry["rho"]["real"], rho_imag=geometry["rho"]["imag"]))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "font.family": "serif", "lines.linewidth": 1.2, "lines.markersize": 3.5})
C = {"raw": "#b03a2e", "phase": "#1f618d", "oracle": "#1e8449", "csi": "#7f7f7f", "p1024": "#5dade2"}
W1 = 3.45

def mean_acc(name):
    agg = collections.defaultdict(list)
    for r in csv.DictReader(open(RUN / f"{name}.csv")):
        agg[(int(r["nr"]), r["spectrum"], round(float(r["d"]), 3), r["method"], r["pilot_variant"])].append(float(r["accuracy"]))
    return {k: 100 * np.mean(v) for k, v in agg.items()}

zeros = jn_zeros(0, 3) / (2 * np.pi)

# Fig. 1: J0 lobes (E1)
e1 = mean_acc("E1_data")
ds = sorted({k[2] for k in e1})
fig, axs = plt.subplots(2, 1, figsize=(W1, 3.3), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
dd = np.linspace(0.3, 1.5, 500)
axs[0].plot(dd, j0(2 * np.pi * dd), color=C["oracle"]); axs[0].axhline(0, color="#999999", lw=0.6)
axs[0].set_ylabel(r"$\rho=J_0(2\pi d)$")
for a in axs:
    for z in zeros: a.axvline(z, color="#cccccc", lw=0.6, ls=":")
for m, lab in [("raw", "uncorrected"), ("phase", "phase-corrected"), ("csi", "CSI")]:
    pv = "0" if m == "csi" else "64"
    axs[1].plot(ds, [e1[(1, "isotropic", d, m, pv)] for d in ds], "-o", color=C[m], label=lab, markersize=3 if m != "csi" else 0)
axs[1].set_xlabel(r"displacement $d$ (wavelengths)"); axs[1].set_ylabel("test accuracy (%)")
axs[1].set_ylim(30, 100); axs[1].legend(frameon=False, loc="lower center", ncol=3)
fig.tight_layout(); fig.savefig(HERE / "fig1_lobes.pdf")

# Fig. 2: dead zone (E2) and delayed return (E3)
e2, e3 = mean_acc("E2_data"), mean_acc("E3_data")
fig, axs = plt.subplots(2, 1, figsize=(W1, 4.2), gridspec_kw={"height_ratios": [1, 1]})
d2 = [0.38, 0.40, 0.55, 0.70]
for (m, pv), ls, mk, col, lab in [(("oracle", "0"), "-", "o", C["oracle"], r"phase, exact $\rho$"),
                                  (("phase", "1024"), "--", "s", C["p1024"], "phase, 1024 pairs"),
                                  (("phase", "64"), ":", "^", C["phase"], "phase, 64 pairs"),
                                  (("raw", "64"), "-", "x", C["raw"], "uncorrected"),
                                  (("csi", "0"), "-", "", C["csi"], "CSI")]:
    axs[0].plot(d2, [e2[(1, "isotropic", d, m, pv)] for d in d2], ls=ls, marker=mk or None, color=col, label=lab)
axs[0].axvline(zeros[0], color="#cccccc", lw=0.6, ls=":")
axs[0].set_xlabel(r"displacement $d$ (wavelengths)"); axs[0].set_ylabel("test accuracy (%)"); axs[0].set_ylim(30, 100)
axs[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.52), ncol=2)
d0s = [0.38, 0.40, 0.42]; arms = [("raw", C["raw"], "uncorrected"), ("phase", C["phase"], "phase"),
                                  ("phase1", "#85c1e9", "1-bit phase"), ("scalar", "#b9770e", "scalar")]
w = 0.18
for i, (m, col, lab) in enumerate(arms):
    axs[1].bar(np.arange(3) + (i - 1.5) * w, [e3[(1, "isotropic", d, m, "64")] for d in d0s], w, color=col, label=f"{lab} at $d_0$")
axs[1].axhline(e3[(1, "isotropic", 0.61, "phase", "64")], color=C["oracle"], lw=1.2, label=r"phase after waiting to $d_1=0.61$")
axs[1].axhline(e3[(1, "isotropic", 0.38, "csi", "0")], color=C["csi"], lw=0.8, ls="--", label="CSI")
axs[1].set_xticks(range(3)); axs[1].set_xticklabels([rf"$d_0={d:.2f}$" for d in d0s])
axs[1].set_ylabel("test accuracy (%)"); axs[1].set_ylim(0, 100)
axs[1].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.tight_layout(); fig.savefig(HERE / "fig2_deadzone.pdf")
print("ok")
