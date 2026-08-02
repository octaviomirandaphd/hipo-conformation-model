"""
HIPO conformation model — figure driver
=======================================
Every figure is produced from hipo_jax_core, so the pairwise joints, the
anchor/Phi_rest joints and the cis/trans profiles cannot disagree.

    python hipo_plots_jax.py
"""
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hipo import core as H

m, dφ, PHI = H.MODEL, H.DPHI, H.PHI_GRID
ALL_NMERS = [3, 4, 5, 10, 15, 20]

BASE_FS = 10
matplotlib.rcParams.update({
    "font.size": BASE_FS*1.5, "axes.titlesize": BASE_FS*1.5,
    "axes.labelsize": BASE_FS*1.5, "xtick.labelsize": BASE_FS*1.5,
    "ytick.labelsize": BASE_FS*1.5, "legend.fontsize": BASE_FS*1.1,
    "figure.titlesize": BASE_FS*1.8,
})

# ── display window: -90..270 so the 180° peaks sit centrally ────────────────
PHI_LO, PHI_HI = -90, 270
_ROLL = -int(round((-90 - PHI[0]) / dφ))
PHI_PLOT = np.where(np.roll(PHI, _ROLL) < -90, np.roll(PHI, _ROLL) + 360,
                    np.roll(PHI, _ROLL))

def roll1(y):  return np.roll(np.asarray(y), _ROLL)
def roll2(z):  return np.roll(np.roll(np.asarray(z), _ROLL, 0), _ROLL, 1)

# ── guide lines at the two coplanar minima ──────────────────────────────────
GUIDE_PHI   = (0, 180)
GUIDE_KW_1D = dict(color="black", linestyle=(0, (5, 4)), linewidth=1.2, zorder=0.5)
GUIDE_KW_2D = dict(color="black", linestyle=(0, (5, 4)), linewidth=1.1,
                   alpha=0.6, zorder=3)

def guides(ax, which="v", style="1d"):
    kw = GUIDE_KW_1D if style == "1d" else GUIDE_KW_2D
    for g in GUIDE_PHI:
        ax.axvline(g, **kw)
        if which == "vh":
            ax.axhline(g, **kw)

LN_FLOOR = -25.0
def ln_clip(z):
    ln = np.log(np.where(z > 1e-300, z, 1e-300))
    return np.where(ln < LN_FLOOR, LN_FLOOR, ln)

def joint_panel(fig, ax, z, title, xlabel, ylabel, cmap, ln=False):
    zz = roll2(z)
    if ln:
        zz, vmin = ln_clip(zz), LN_FLOOR
    else:
        vmin = None
    im = fig.colorbar(ax.pcolormesh(PHI_PLOT, PHI_PLOT, zz.T, shading="auto",
                                    cmap=cmap, vmin=vmin), ax=ax)
    im.set_ticks([])
    guides(ax, "vh", "2d")
    ax.set_title(title, pad=20)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(PHI_LO, PHI_HI); ax.set_ylim(PHI_LO, PHI_HI)
    ax.set_xticks(range(-90, 271, 90)); ax.set_yticks(range(-90, 271, 90))

_n = [0]
def _fn(stub):
    _n[0] += 1
    return f"fig{_n[0]:02d}_{stub}.png"

OUTDIR = Path(__file__).resolve().parents[2] / "figures"
OUTDIR.mkdir(exist_ok=True)

def save(fig, stub):
    f = OUTDIR / _fn(stub)
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved figures/{f.name}")


# ═══ Figure set 1 — P(phi|T) ════════════════════════════════════════════════
def fig_tautomers():
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (nm, c, mk) in enumerate([("Ta", "#D32F2F", "D"), ("Tb", "#F9A825", "o"),
                                     ("Tc", "#1565C0", "^"), ("Td", "#2E7D32", "v")]):
        y = roll1(m.P_phi_T[i])
        ax.plot(PHI_PLOT, y, color=c, lw=2, label=f"{nm}  [P(T) = {float(m.P_T[i]):.4f}]")
        step = int(15/dφ)
        ax.plot(PHI_PLOT[::step], y[::step], mk, color=c, ls="none", ms=8)
    ax.set_title("P(φ | T)", pad=20)
    ax.set_xlabel("φ (°)"); ax.set_ylabel("P(φ | T)")
    ax.set_xlim(PHI_LO, PHI_HI); ax.set_xticks(range(-90, 271, 45))
    ax.axhline(0, color="k", lw=0.4); guides(ax); ax.legend()
    fig.tight_layout(); save(fig, "tautomer_distributions")


# ═══ Figure set 2 — pairwise joints + marginals ═════════════════════════════
def fig_pairwise(n, normalised=True, ln=False):
    N = n - 1
    tag = ("norm" if normalised else "unnorm") + ("_ln" if ln else "_linear")
    jf = np.asarray(H.pairwise_joint(m, 0, N-1, "forward", normalised))
    jr = np.asarray(H.pairwise_joint(m, 0, N-1, "reverse", normalised))
    ja = 0.5*(jf+jr)
    if normalised:
        ja = ja/(ja.sum()*dφ**2)

    SEL = [k for k in [0, 1, 2, 3, 8, 13, 18] if k < N]
    cv = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 19))
    cp = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, 19))

    fig = plt.figure(figsize=(19, 14))
    fig.suptitle(f"{n}-mer — {'normalised' if normalised else 'unnormalised'}"
                 f"{' (ln scale)' if ln else ''}   N={N} dimers",
                 fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36)

    pre = "ln P" if ln else "P"
    for col, (z, ttl, cm_) in enumerate([
            (jf, f"{pre}(φ1,φ{N}) — {H.LABEL['forward']}",  "viridis"),
            (jr, f"{pre}(φ1,φ{N}) — {H.LABEL['reverse']}",  "viridis"),
            (ja, f"{pre}(φ1,φ{N}) — Orientation-averaged",  "plasma")]):
        joint_panel(fig, fig.add_subplot(gs[0, col]), z, ttl,
                    "φ1 (°)", f"φ{N} (°)", cm_, ln)

    ymax = float(np.asarray(m.P_phi_T).max())*1.05
    series = [
        ([np.asarray(H.marginal(m, k, "forward", normalised)) for k in SEL], cv, H.LABEL["forward"]),
        ([np.asarray(H.marginal(m, k, "reverse", normalised)) for k in SEL], cv, H.LABEL["reverse"]),
        ([np.asarray(H.avg_marginal(m, n, k, normalised))     for k in SEL], cp, "Orientation-averaged"),
    ]
    for col, (ys, cols, ttl) in enumerate(series):
        ax = fig.add_subplot(gs[1, col])
        lines = [ax.plot(PHI_PLOT, roll1(y), color=cols[k], lw=2, label=f"φ{k+1}")[0]
                 for y, k in zip(ys, SEL)]
        ax.set_title(f"P(φ) — {ttl}", pad=20)
        ax.set_xlabel("φ (°)")
        ax.set_xlim(PHI_LO, PHI_HI); ax.set_xticks(range(-90, 271, 45))
        ax.set_ylim(0, ymax*1.18); ax.set_yticklabels([])
        ax.axhline(0, color="k", lw=0.4); guides(ax)
        ax.legend(handles=lines, ncol=max(1, len(SEL)//3+1), loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), framealpha=0.9,
                  columnspacing=0.8, handlelength=1.4)
    save(fig, f"{tag}_{n}mer")


# ═══ Figure set 3 — anchor / Phi_rest joints ════════════════════════════════
def fig_anchor(n):
    N = n - 1
    jf = np.asarray(H.anchor_phirest_joint(m, n, "first", "forward"))
    jl = np.asarray(H.anchor_phirest_joint(m, n, "last",  "forward"))
    lab_f = f"φ2+…+φ{N}" if N > 2 else "φ2"
    lab_l = f"φ1+…+φ{N-1}" if N > 2 else "φ1"

    fig = plt.figure(figsize=(13, 12))
    fig.suptitle(f"{n}-mer — P(φ_anchor, Φ_rest)   N={N} dimers", fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    for (z, a_sym, r_sym, r, c, ln) in [
            (jl, f"φ{N}", lab_l, 0, 0, False), (jf, "φ1", lab_f, 0, 1, False),
            (jl, f"φ{N}", lab_l, 1, 0, True),  (jf, "φ1", lab_f, 1, 1, True)]:
        joint_panel(fig, fig.add_subplot(gs[r, c]), z,
                    f"{'ln P' if ln else 'P'}({a_sym}, Φ_rest)",
                    f"{a_sym} (°)", f"Φ_rest = {r_sym} (°)",
                    "plasma" if ln else "viridis", ln)
    save(fig, f"anchor_phirest_{n}mer")


# ═══ Figure set 4 — cis/trans profile along the chain ═══════════════════════
def fig_cis_trans():
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("Orientation-averaged cis/trans profile along the chain",
                 fontweight="bold")
    for ax, n in zip(axes.ravel(), ALL_NMERS):
        tr, ci = H.cis_trans_profile(m, n)
        k = np.arange(1, len(tr)+1)
        ax.plot(k, tr, "o-", color="#D32F2F", lw=2, ms=7, label="trans (0±30°)")
        ax.plot(k, ci, "s-", color="#2E7D32", lw=2, ms=7, label="cis (180±30°)")
        ax.plot(k, 1-tr-ci, "^:", color="0.5", lw=1.5, ms=6, label="other")
        ax.set_title(f"n = {n}"); ax.set_xlabel("dimer index k")
        ax.set_ylabel("probability"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.legend(fontsize=BASE_FS)
    fig.tight_layout(); save(fig, "cis_trans_profile")


if __name__ == "__main__":
    print("Figures:")
    fig_tautomers()
    for n in ALL_NMERS:
        for norm in (True, False):
            for ln in (False, True):
                fig_pairwise(n, norm, ln)
    for n in ALL_NMERS:
        fig_anchor(n)
    fig_cis_trans()

    print("\nΦ_rest cis/trans (corrected):")
    mt, mc = np.asarray(H.MASK_TRANS), np.asarray(H.MASK_CIS)
    print(f"  {'n':>4} {'trans':>9} {'cis':>9}")
    for n in ALL_NMERS:
        J = np.asarray(H.anchor_phirest_joint(m, n, "first", "forward"))
        r = J.sum(0)*dφ; r /= r.sum()*dφ
        print(f"  {n:>4} {r[mt].sum()*dφ:>9.4f} {r[mc].sum()*dφ:>9.4f}")
