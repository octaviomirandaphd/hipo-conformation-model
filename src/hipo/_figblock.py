# ==============================================================================
# PUBLISHED FIGURE SUITE
# ==============================================================================
# One function per published figure, named for the figure it produces.  This
# block is the SINGLE SOURCE for figure code: it is inserted verbatim into both
# src/hipo/plots.py and hipo_model_standalone.py, and tests/test_figure_parity.py
# asserts the two copies are byte-identical.  Do not edit one without the other.
#
# Every function takes (outdir) and returns the path it wrote.
# ------------------------------------------------------------------------------
try:                                   # packaged use: hipo.plots imports this
    from hipo.core import *             # noqa: F401,F403
except ImportError:                     # standalone use: names already in scope
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from pathlib import Path

_BASE_FS = 10
matplotlib.rcParams.update({
    "font.size": _BASE_FS * 1.5, "axes.titlesize": _BASE_FS * 1.5,
    "axes.labelsize": _BASE_FS * 1.5, "xtick.labelsize": _BASE_FS * 1.5,
    "ytick.labelsize": _BASE_FS * 1.5, "legend.fontsize": _BASE_FS * 1.1,
    "figure.titlesize": _BASE_FS * 1.8})

ALL_NMERS = [3, 4, 5, 10, 15, 20]          # chain lengths used in S3
S4_NMERS  = [3, 4, 5, 6, 10, 15]           # chain lengths used in S4

# tautomer colours, shared by every figure
C_TA, C_TB, C_TC, C_TD = "#D32F2F", "#F9A825", "#1565C0", "#2E7D32"
C_GLOBAL = "#4A148C"
C_ANCHOR_F, C_ANCHOR_R = "#7B2D8B", "#F5A800"   # S3 panel f / Fig 4d-e accents

# display window: -90..270 so the 180 deg features sit centrally
_LO, _HI = -90, 270
_ROLL = -int(round((-90 - PHI_GRID[0]) / DPHI))
PHI_PLOT = np.where(np.roll(PHI_GRID, _ROLL) < -90,
                    np.roll(PHI_GRID, _ROLL) + 360, np.roll(PHI_GRID, _ROLL))


def _r1(y):
    return np.roll(np.asarray(y, dtype=np.float64), _ROLL)


def _r2(z):
    z = np.asarray(z, dtype=np.float64)
    return np.roll(np.roll(z, _ROLL, 0), _ROLL, 1)


_GK1 = dict(color="black", linestyle=(0, (5, 4)), linewidth=1.2, zorder=0.5)
_GK2 = dict(color="black", linestyle=(0, (5, 4)), linewidth=1.1, alpha=0.6, zorder=3)


def _guides(ax, which="v"):
    kw = _GK1 if which == "v" else _GK2
    for g in (0, 180):
        ax.axvline(g, **kw)
        if which == "vh":
            ax.axhline(g, **kw)


def _style_x(ax, step=45):
    ax.set_xlabel("φ (°)")
    ax.set_xlim(_LO, _HI)
    ax.set_xticks(range(-90, 271, step))
    ax.axhline(0, color="k", lw=0.4)


_LN_FLOOR = -25.0


def _joint_panel(fig, ax, z, title, xl, yl, cmap, ln=False):
    zz = _r2(z)
    if ln:
        zz = np.log(np.where(zz > 1e-300, zz, 1e-300))
        zz = np.where(zz < _LN_FLOOR, _LN_FLOOR, zz)
        vmin = _LN_FLOOR
    else:
        vmin = None
    cb = fig.colorbar(ax.pcolormesh(PHI_PLOT, PHI_PLOT, zz.T, shading="auto",
                                    cmap=cmap, vmin=vmin), ax=ax)
    cb.set_ticks([])
    _guides(ax, "vh")
    ax.set_title(title, pad=20)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_xlim(_LO, _HI); ax.set_ylim(_LO, _HI)
    ax.set_xticks(range(-90, 271, 90)); ax.set_yticks(range(-90, 271, 90))


def _save(fig, outdir, name):
    out = Path(outdir); out.mkdir(exist_ok=True, parents=True)
    f = out / name
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {f}")
    return f


# ------------------------------------------------------------------------------
# Chain-position convention for the profile figures (4f, S4)
# ------------------------------------------------------------------------------
# The model propagates from the T_a-rich end; the paper indexes from the
# T_d-rich end.  taut_profile() and cis_trans_profile() return arrays in MODEL
# order (index 0 = T_a-rich).  The published figures plot paper position
#     k_paper = N - k_model,
# i.e. the same arrays against a reversed axis, so k = 1 is the T_d/cis head and
# k = N is the T_a/trans tail.  This is the LABEL inversion declared in core,
# applied to the chain axis instead of the plot title.
# ------------------------------------------------------------------------------
def _paper_positions(N):
    """Paper dimer positions for model indices 0 .. N-1.

    Routed through core.to_paper_index so the conversion has exactly one
    definition.  It previously duplicated the arithmetic inline, which left
    to_paper_index with no caller anywhere in the repository -- a mutation that
    made it return nonsense was invisible.
    """
    return np.array([to_paper_index(N + 1, k) for k in range(N)])


def _profile_panel(m, ax, n, delta=0.10, show_legend=False, label=None):
    N = n - 1
    x = _paper_positions(N)
    g = taut_profile(m, n)
    tr, ci = cis_trans_profile(m, n)

    ax.plot(x, g[:, 0], color=C_TA, lw=1.8, marker="D", ms=6)
    ax.plot(x, g[:, 2], color=C_TC, lw=1.8, marker="^", ms=6)
    ax.plot(x, g[:, 3], color=C_TD, lw=1.8, marker="v", ms=6)
    ax.plot(x, tr, color=C_TA, lw=1.8, ls="--", marker="D", ms=6,
            markerfacecolor="none", markeredgewidth=1.2)
    ax.plot(x, ci, color=C_TD, lw=1.8, ls="--", marker="v", ms=6,
            markerfacecolor="none", markeredgewidth=1.2)

    # T_c bridge region: eq 21, |gamma(Ta) - gamma(Td)| < delta.
    # Only drawn for n >= 10; on shorter chains the criterion is either empty or
    # spans most of the chain, so shading it would overstate the case.
    reg = tc_bridge_region(m, n, delta)
    if n >= 10 and reg is not None:
        lo, hi = reg
        xa, xb = N - hi, N - lo                      # model -> paper positions
        ax.axvspan(xa - 0.5, xb + 0.5, alpha=0.12, color=C_TC, zorder=0)
        ax.annotate("Tc bridge\nregion", xy=(xa - 0.2, 0.93),
                    xycoords=("data", "axes fraction"), fontsize=_BASE_FS,
                    color=C_TC, ha="left", va="top")

    # k*: computed from the second eigenvalues, not hardcoded.
    ks_model, _ = k_star(m, n)
    ks_paper = N - ks_model + 1
    ax.axvline(ks_paper, color=C_TC, lw=1.2, ls="--", alpha=0.85)
    ax.annotate(f"k*={ks_paper:.1f}", xy=(ks_paper, 0.20),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=_BASE_FS * 1.1,
                color=C_TC, va="bottom", ha="left")

    ax.axhline(0, color="k", lw=0.4)
    ax.set_title(label or f"n = {n}  (N = {N} dimers)", pad=20)
    ax.set_xlabel("Dimer position k")
    ax.set_ylabel("Probability")
    ax.set_xlim(0.5, N + 0.5)
    ax.set_xticks(np.arange(1, N + 1) if N <= 10 else np.arange(1, N + 1, 2))
    ax.set_ylim(0, 1)
    # No head/tail annotations: the legend already gives Ta/Td and cis/trans,
    # and the curves themselves identify which end is which.
    if show_legend:
        h = [mlines.Line2D([], [], color=C_TA, lw=1.8, marker="D", ms=6, label="Ta"),
             mlines.Line2D([], [], color=C_TC, lw=1.8, marker="^", ms=6, label="Tc"),
             mlines.Line2D([], [], color=C_TD, lw=1.8, marker="v", ms=6, label="Td"),
             mlines.Line2D([], [], color=C_TA, lw=1.8, ls="--", marker="D", ms=6,
                           mfc="none", mew=1.2, label="trans"),
             mlines.Line2D([], [], color=C_TC, lw=1.2, ls="--", label="k* (Tc bridge)"),
             mlines.Line2D([], [], color=C_TD, lw=1.8, ls="--", marker="v", ms=6,
                           mfc="none", mew=1.2, label="cis")]
        ax.legend(handles=h, loc="upper right", ncol=2, fontsize=_BASE_FS * 0.95)


# ==============================================================================
# MAIN TEXT
# ==============================================================================
def fig_1c(m, outdir="figures"):
    """Figure 1c — relative Gibbs free energy profile of the four IPD tautomers."""
    dgphi = np.asarray(m.dG_phi, dtype=np.float64)
    dg = np.asarray(m.dG, dtype=np.float64)
    x = np.where(dgphi < -90, dgphi + 360, dgphi)
    o = np.argsort(x)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (nm, c, mk) in enumerate([("Ta", C_TA, "D"), ("Tb", C_TB, "o"),
                                     ("Tc", C_TC, "^"), ("Td", C_TD, "v")]):
        ax.plot(x[o], dg[i][o], color=c, lw=2, marker=mk, ms=7, label=nm)
    ax.plot(x[o], dg[4][o], color="0.35", lw=2, ls="--", label="Boltzmann average")
    ax.set_title("ΔG torsional profile — IPD tautomers", pad=20)
    ax.set_ylabel("ΔG (kcal mol⁻¹)")
    _style_x(ax)
    _guides(ax)
    ax.legend(ncol=2)
    fig.tight_layout()
    return _save(fig, outdir, "fig_1c_dG_torsional_profile.png")


def fig_3a(m, outdir="figures"):
    """Figure 3a — conditional P(φ|T) for the four tautomers plus global P(φ)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    step = int(15 / DPHI)
    for i, (nm, c, mk) in enumerate([("Ta", C_TA, "D"), ("Tb", C_TB, "o"),
                                     ("Tc", C_TC, "^"), ("Td", C_TD, "v")]):
        y = _r1(m.P_phi_T[i])
        ax.plot(PHI_PLOT, y, color=c, lw=2,
                label=f"{nm}  [P(T) = {float(m.P_T[i]):.4f}]")
        ax.plot(PHI_PLOT[::step], y[::step], mk, color=c, ls="none", ms=8)
    ax.plot(PHI_PLOT, _r1(global_marginal(m)), color=C_GLOBAL, lw=2.5, ls="--",
            label="P(φ) global  (eq 4)")
    ax.set_title("P(φ | T) and the global torsional distribution", pad=20)
    ax.set_ylabel("Probability density")
    _style_x(ax)
    _guides(ax)
    ax.legend()
    fig.tight_layout()
    return _save(fig, outdir, "fig_3a_conditional_and_global.png")


def fig_4abc(m, outdir="figures", n=20, ln=False, normalised=True):
    """Figure 4a-c — pairwise joint distributions for the n-mer.

    a) reverse-direction, b) forward-direction, c) orientation-averaged.
    Panel c uses orientation_averaged_joint (reverse term transposed).
    """
    N = n - 1
    jf = pairwise_joint(m, 0, N - 1, "forward", normalised)
    jr = pairwise_joint(m, 0, N - 1, "reverse", normalised)
    ja = orientation_averaged_joint(m, 0, N - 1, normalised)
    pre = "ln P" if ln else "P"

    # PANEL ORDER AND ORIENTATION -- read this before editing.
    #
    # `pairwise_joint` always places the walk's PRIOR on axis 0, in both
    # directions, so axis 0 is a different physical dimer in each:
    #     jf (code "forward",  funnels to Td): axis0 = paper phi_N, axis1 = phi_1
    #     jr (code "reverse",  funnels to Ta): axis0 = paper phi_1, axis1 = phi_N
    # The manuscript captions 4a as the reverse direction, which in its own
    # naming is [T_R] = code T_fwd = jf, i.e. the Td-funnelling panel.  So a
    # takes jf and b takes jr -- the opposite of what this function used to do.
    #
    # Every panel then plots x = paper phi_1 and y = paper phi_N, so a reader
    # sees the same physical dimer on the same axis throughout.  That needs a
    # transpose on the two panels whose axis 0 is paper phi_N (a and c); b is
    # already in that orientation.  A transpose is a horizontal flip plus a
    # 90-degree rotation, which is the manual edit that was previously applied
    # to the published figure by hand.
    a_panel = jf.T          # reverse direction, Td-funnelling
    b_panel = jr            # forward direction, Ta-funnelling
    c_panel = ja.T          # orientation-averaged, shares jf's axis convention
    fig = plt.figure(figsize=(19, 6.5))
    fig.suptitle(f"Pairwise joint distributions — {n}-mer   (N = {N} dimers)"
                 f"{'' if normalised else ', unnormalised'}"
                 f"{' , ln scale' if ln else ''}", fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.36)
    for col, (z, ttl, cm) in enumerate([
            (a_panel, f"a)  {pre}(φ1,φ{N}) — reverse direction", "viridis"),
            (b_panel, f"b)  {pre}(φ1,φ{N}) — forward direction", "viridis"),
            (c_panel, f"c)  {pre}(φ1,φ{N}) — orientation-averaged", "plasma")]):
        _joint_panel(fig, fig.add_subplot(gs[0, col]), z, ttl,
                     "φ1 (°)", f"φ{N} (°)", cm, ln)
    tag = ("norm" if normalised else "unnorm") + ("_ln" if ln else "_linear")
    return _save(fig, outdir, f"fig_4abc_pairwise_joints_{n}mer_{tag}.png")


def fig_4de(m, outdir="figures", n=20, ln=False):
    """Figure 4d-e — cumulative torsional joints P(φ_anchor, Φ_rest).

    d) forward direction, φ1 anchor.   e) reverse direction, φN anchor.
    """
    N = n - 1
    jf = anchor_phirest_joint(m, n, "last", "forward")
    jr = anchor_phirest_joint(m, n, "last", "reverse")
    # The two panels hold out DIFFERENT anchors, so their Phi sums differ:
    # d) anchors phi_1  -> Phi = phi_2 + ... + phi_N
    # e) anchors phi_N  -> Phi = phi_1 + ... + phi_{N-1}
    # A single shared label printed the d) definition on both panels.
    lab_d = f"φ2+…+φ{N}" if N > 2 else "φ2"
    lab_e = f"φ1+…+φ{N-1}" if N > 2 else "φ1"
    pre = "ln P" if ln else "P"
    fig = plt.figure(figsize=(13, 6.5))
    fig.suptitle(f"Cumulative torsional mapping — {n}-mer   (N = {N} dimers)",
                 fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    for col, (z, ttl, lb, ax_lab) in enumerate([
            (jf, f"d)  {pre}(φ1, Φ) — forward, φ1 anchor", lab_d, "φ1"),
            (jr, f"e)  {pre}(φ{N}, Φ) — reverse, φ{N} anchor", lab_e, f"φ{N}")]):
        _joint_panel(fig, fig.add_subplot(gs[0, col]), z, ttl,
                     f"{ax_lab} (°)", f"Φ = {lb} (°)",
                     "plasma" if ln else "viridis", ln)
    return _save(fig, outdir,
                 f"fig_4de_cumulative_joints_{n}mer{'_ln' if ln else ''}.png")


def fig_4f(m, outdir="figures", n=20, delta=0.10):
    """Figure 4f — tautomer and cis/trans profile along the 20-mer chain,
    with the k* marker and the T_c bridge region."""
    fig, ax = plt.subplots(figsize=(11, 7))
    _profile_panel(m, ax, n, delta, show_legend=True,
                   label=f"f)  {n}-mer conformational profile  (N = {n-1} dimers)")
    fig.tight_layout()
    return _save(fig, outdir, f"fig_4f_profile_{n}mer.png")


# ==============================================================================
# SUPPORTING INFORMATION
# ==============================================================================
def fig_S1(m, outdir="figures"):
    """Figure S1 — ΔG and Boltzmann probability per tautomer, panels a-d."""
    dgphi = np.asarray(m.dG_phi, dtype=np.float64)
    dg = np.asarray(m.dG, dtype=np.float64)
    xr = np.where(dgphi < -90, dgphi + 360, dgphi)
    o = np.argsort(xr)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("ΔG and Boltzmann probability distributions per IPD tautomer",
                 fontweight="bold")
    for ax, i, nm, c, lab in zip(axes.ravel(), range(4),
                                 ["Ta", "Tb", "Tc", "Td"],
                                 [C_TA, C_TB, C_TC, C_TD], "abcd"):
        ax.plot(xr[o], dg[i][o], color=c, lw=2, marker="o", ms=6, label="ΔG")
        ax.set_ylabel("ΔG (kcal mol⁻¹)", color=c)
        ax.tick_params(axis="y", labelcolor=c)
        ax.set_title(f"{lab})  {nm}   [P(T) = {float(m.P_T[i]):.4g}]", pad=20)
        _style_x(ax, step=90)
        _guides(ax)
        ax2 = ax.twinx()
        ax2.plot(PHI_PLOT, _r1(m.P_phi_T[i]), color="0.35", lw=2, ls="--",
                 label="P(φ|T)")
        ax2.set_ylabel("P(φ | T)", color="0.35")
        ax2.tick_params(axis="y", labelcolor="0.35")
        ax2.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, outdir, "fig_S1_dG_and_probability.png")


def fig_S3(m, outdir="figures", n_detail=20, nmers=None):
    """
    Figure S3 — marginal distributions across chain lengths.

    a) dimer k1 (Td head)          b) dimer k_{n-1} (Ta tail)
    c) orientation-averaged        d) Φ forward (k1 anchor)
    e) Φ reverse (k_{n-1} anchor)  f) overlay for the n_detail-mer

    Every curve is a CHAIN LENGTH, so every legend entry is n=... .

    Panel c uses the manuscript eq-6 mirrored index,
        P_avg(k) = 1/2 [ P_fwd(k) + P_rev(N-1-k) ].
    The earlier standalone script for this figure used the SAME index for both
    directions.  That makes the k=0 curve equal to the prior for every chain
    length -- a whole family of curves collapsing onto one line, independent of
    n -- and it disagrees with eq 6 by 5.3e-03.  See the note in
    orientation_averaged_joint: same convention error, marginal instead of joint.
    """
    nmers = list(nmers or ALL_NMERS)
    cv = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 19))
    cp = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, 19))

    head, tail, avg_head, avg_tail = [], [], [], []
    rest_f, rest_r, anc_f, anc_r = {}, {}, {}, {}
    for n in nmers:
        N = n - 1
        head.append(np.asarray(marginal(m, N - 1, "forward")))   # Td/cis head
        tail.append(np.asarray(marginal(m, N - 1, "reverse")))   # Ta/trans tail
        avg_head.append(np.asarray(avg_marginal(m, n, N - 1)))
        avg_tail.append(np.asarray(avg_marginal(m, n, 0)))
        for dr, ra, aa in (("forward", rest_f, anc_f), ("reverse", rest_r, anc_r)):
            J = np.asarray(anchor_phirest_joint(m, n, "last", dr, True),
                           dtype=np.float64)
            a = J.sum(1) * DPHI; a /= a.sum() * DPHI
            r = J.sum(0) * DPHI; r /= r.sum() * DPHI
            aa[n] = a; ra[n] = r

    ymax = max(float(np.asarray(y).max())
               for y in head + tail + avg_head + avg_tail) * 1.62
    # 1.62 is the smallest headroom that clears panel c's two stacked legends,
    # the lower of which is a single column so it reads vertically like the ones
    # in a, b, d and e.  Going higher flattens the peaks in all three panels for
    # no gain.  Applied to
    # all three top panels so a, b and c stay on an identical y-scale and
    # peak heights remain directly comparable across them.
    Nd = n_detail - 1

    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    fig.suptitle("Marginal distributions across chain lengths", fontweight="bold")

    def _leg(ax):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  fontsize=_BASE_FS * 1.1, framealpha=0.9,
                  columnspacing=0.8, handlelength=1.4)

    ax = axes[0, 0]
    for y, n in zip(head, nmers):
        ax.plot(PHI_PLOT, _r1(y), color=cv[n - 2], lw=2, label=f"n={n}")
    ax.set_title("a)  P(φ1) — dimer k₁ (Td/cis head)", pad=20)
    ax.set_ylabel("Probability density"); ax.set_ylim(0, ymax)
    _style_x(ax, 90); _guides(ax); _leg(ax)

    ax = axes[0, 1]
    for y, n in zip(tail, nmers):
        ax.plot(PHI_PLOT, _r1(y), color=cv[n - 2], lw=2, label=f"n={n}")
    ax.set_title("b)  P(φ_N) — dimer k_{n−1} (Ta/trans tail)", pad=20)
    ax.set_ylim(0, ymax); ax.set_yticklabels([])
    _style_x(ax, 90); _guides(ax); _leg(ax)

    # Panel c carries TWO curves per chain length, because the orientation-averaged
    # marginal is not the same at the two ends:
    #     avg(k)   = 1/2 [ fwd(k) + rev(N-1-k) ]      (manuscript eq 6)
    #     avg(0)   = 1/2 [ prior + rev converged to Ta ]
    #     avg(N-1) = 1/2 [ fwd converged to Td + prior ]
    # Those differ (1.15e-02 at n = 20) because [T_F] is not the time-reverse of
    # [T_R], so the averaged profile is NOT symmetric under k -> N-1-k.  Panels a
    # and b each show one end and need one curve each; panel c is the averaged
    # counterpart of BOTH, so it needs two.  Solid = head, dashed = tail, and the
    # second legend says so -- without it the panel looks like a duplication bug.
    ax = axes[0, 2]
    for y1, y2, n in zip(avg_head, avg_tail, nmers):
        ax.plot(PHI_PLOT, _r1(y1), color=cp[n - 2], lw=2.2, ls="-", label=f"n={n}")
        ax.plot(PHI_PLOT, _r1(y2), color=cp[n - 2], lw=2.2, ls=(0, (6, 3)))
    ax.set_title("c)  orientation-averaged, both ends", pad=20)
    ax.set_ylim(0, ymax); ax.set_yticklabels([])
    _style_x(ax, 90); _guides(ax)
    # Two stacked, centred legends: the line-style key on top (which end of the
    # chain), the chain-length key beneath it (which colour).  ax.legend()
    # replaces any previous legend, so the first is re-added as an artist.  The
    # headroom for both comes from the shared ymax set above.
    _end_legend = ax.legend(
        handles=[mlines.Line2D([], [], color="0.25", lw=2.2, ls="-",
                               label="k$_1$ (Td head)"),
                 mlines.Line2D([], [], color="0.25", lw=2.2, ls=(0, (6, 3)),
                               label="k$_{n-1}$ (Ta tail)")],
        loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2,
        fontsize=_BASE_FS * 0.85, framealpha=0.9, handlelength=2.6,
        columnspacing=0.9, labelspacing=0.25, borderpad=0.35)
    ax.add_artist(_end_legend)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=1,
              fontsize=_BASE_FS * 0.95, framealpha=0.9,
              columnspacing=0.8, handlelength=1.4, labelspacing=0.18,
              borderpad=0.35)

    ax = axes[1, 0]
    for n in nmers:
        ax.plot(PHI_PLOT, _r1(rest_f[n]), color=cv[n - 2], lw=2, label=f"n={n}")
    ax.set_title("d)  Φ = φ2+…+φ_N   (forward, k₁ anchor)", pad=20)
    ax.set_ylabel("Probability density  P(Φ)"); ax.set_ylim(bottom=0)
    _style_x(ax, 90); _guides(ax); _leg(ax)

    ax = axes[1, 1]
    for n in nmers:
        ax.plot(PHI_PLOT, _r1(rest_r[n]), color=cv[n - 2], lw=2, label=f"n={n}")
    ax.set_title("e)  Φ = φ1+…+φ_{N−1}   (reverse, k_{n−1} anchor)", pad=20)
    ax.set_ylim(bottom=0); ax.set_yticklabels([])
    _style_x(ax, 90); _guides(ax); _leg(ax)

    ax = axes[1, 2]
    ax.plot(PHI_PLOT, _r1(anc_f[n_detail]), color=C_ANCHOR_F, lw=2, label="φ1")
    ax.plot(PHI_PLOT, _r1(rest_f[n_detail]), color=C_ANCHOR_F, lw=2, ls="--",
            label="Φ$_F$")
    ax.plot(PHI_PLOT, _r1(anc_r[n_detail]), color=C_ANCHOR_R, lw=2,
            label=f"φ{Nd}")
    ax.plot(PHI_PLOT, _r1(rest_r[n_detail]), color=C_ANCHOR_R, lw=2, ls="--",
            label="Φ$_R$")
    ax.set_title(f"f)  n={n_detail}  |  anchors vs cumulative Φ", pad=20)
    ax.set_ylabel("Probability density"); ax.set_ylim(bottom=0)
    _style_x(ax, 90); _guides(ax)
    # Two columns of two, symbols only: the spelled-out "Φ forward"/"Φ reverse"
    # made this legend wide enough to sit on top of the φ1 peak.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=_BASE_FS * 1.1, framealpha=0.9,
              columnspacing=1.2, handlelength=1.6, labelspacing=0.3)

    fig.tight_layout()
    return _save(fig, outdir, "fig_S3_marginals_across_chain_lengths.png")


def fig_S4(m, outdir="figures", nmers=None, delta=0.10):
    """Figure S4 — tautomer and cis/trans profiles for n = 3, 4, 5, 6, 10, 15,
    with the k* marker and, for n >= 10, the T_c bridge region."""
    nmers = list(nmers or S4_NMERS)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("Tautomer and cis/trans probability profiles along the chain\n"
                 "P(trans) = P(φ within ±30° of 0°)    "
                 "P(cis) = P(φ within ±30° of 180°)", fontweight="bold")
    for ax, n, lab in zip(axes.ravel(), nmers, "abcdef"):
        _profile_panel(m, ax, n, delta, show_legend=(lab == "a"),
                       label=f"{lab})  n = {n}   (N = {n-1} dimers)")
    fig.tight_layout()
    return _save(fig, outdir, "fig_S4_profiles_by_chain_length.png")


def fig_S5(m, outdir="figures", n_max=50):
    """Figure S5 — a) convergence of chain-averaged cis/trans with chain length,
    b) saturation of the terminal Ta and Td beliefs at opposing chain ends."""
    # Pure-numpy path.  avg_marginal and belief_at are jitted on static (n, k),
    # so sweeping n = 3..50 through them would trigger ~1200 recompilations.
    # The beliefs are just repeated 4x4 matrix-vector products; precompute each
    # power sequence once and reuse.  Verified against taut_profile below.
    Tf = np.asarray(m.T_fwd, dtype=np.float64)
    Tr = np.asarray(m.T_rev, dtype=np.float64)
    Pp = np.asarray(m.P_phi_T, dtype=np.float64)
    P0 = np.asarray(m.P_T, dtype=np.float64)
    mt, mc = np.asarray(MASK_TRANS), np.asarray(MASK_CIS)

    bf = [P0.copy()]
    br = [P0.copy()]
    for _ in range(n_max):
        bf.append(bf[-1] @ Tf)
        br.append(br[-1] @ Tr)

    ns = np.arange(3, n_max + 1)
    tr_avg, ci_avg, ta_end, td_end = [], [], [], []
    for n in ns:
        N = n - 1
        g = np.array([0.5 * (bf[k] + br[N - 1 - k]) for k in range(N)])
        g /= g.sum(1, keepdims=True)
        y = g @ Pp
        y /= (y.sum(1, keepdims=True) * DPHI)
        tr_avg.append(float((y[:, mt].sum(1) * DPHI).mean()))
        ci_avg.append(float((y[:, mc].sum(1) * DPHI).mean()))
        ta_end.append(float(g[0, 0]))    # model index 0     = Ta/trans tail
        td_end.append(float(g[-1, 3]))   # model index N - 1 = Td/cis head

    # cross-check the fast path against the jitted one at a single chain length
    _g = taut_profile(m, 20)
    _y0, _y1 = np.array([0.5 * (bf[k] + br[18 - k]) for k in range(19)]), _g
    _y0 /= _y0.sum(1, keepdims=True)
    assert np.abs(_y0 - _y1).max() < 1e-12, "S5 fast path disagrees with taut_profile"
    tr_avg, ci_avg = np.array(tr_avg), np.array(ci_avg)
    ta_end, td_end = np.array(ta_end), np.array(td_end)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    fig.suptitle("Convergence with chain length", fontweight="bold")

    ax = axes[0]
    ax.plot(ns, tr_avg, "-", color=C_TA, lw=2.2, label="⟨trans⟩")
    ax.plot(ns, ci_avg, "-", color=C_TD, lw=2.2, label="⟨cis⟩")
    ax.plot(ns, 1 - tr_avg - ci_avg, ":", color="0.5", lw=1.8, label="⟨other⟩")
    ax.axhline(tr_avg[-1], color=C_TA, lw=1.0, ls="--", alpha=0.6)
    ax.axhline(ci_avg[-1], color=C_TD, lw=1.0, ls="--", alpha=0.6)
    ax.annotate(f"{ci_avg[-1]:.3f}", xy=(ns[-1], ci_avg[-1]), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=C_TD,
                fontsize=_BASE_FS * 1.1)
    ax.annotate(f"{tr_avg[-1]:.3f}", xy=(ns[-1], tr_avg[-1]), xytext=(-6, -14),
                textcoords="offset points", ha="right", color=C_TA,
                fontsize=_BASE_FS * 1.1)
    ax.set_title("a)  chain-averaged cis/trans", pad=20)
    ax.set_xlabel("chain length n"); ax.set_ylabel("probability")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(ns, td_end, "-", color=C_TD, lw=2.2, marker="v", ms=4,
            markevery=4, label="Td at the Td/cis head")
    ax.plot(ns, ta_end, "-", color=C_TA, lw=2.2, marker="D", ms=4,
            markevery=4, label="Ta at the Ta/trans tail")
    ax.axhline(td_end[-1], color=C_TD, lw=1.0, ls="--", alpha=0.6)
    ax.axhline(ta_end[-1], color=C_TA, lw=1.0, ls="--", alpha=0.6)
    ax.annotate(f"{ta_end[-1]:.4f}", xy=(ns[-1], ta_end[-1]), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=C_TA,
                fontsize=_BASE_FS * 1.1)
    ax.annotate(f"{td_end[-1]:.4f}", xy=(ns[-1], td_end[-1]), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=C_TD,
                fontsize=_BASE_FS * 1.1)
    ax.set_title("b)  terminal tautomer belief", pad=20)
    ax.set_xlabel("chain length n"); ax.set_ylabel("belief γ(T)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    return _save(fig, outdir, "fig_S5_convergence.png")


# ------------------------------------------------------------------------------
def run_figures(m, outdir="figures", n=20):
    """Write the complete published figure set."""
    fig_1c(m, outdir)
    fig_3a(m, outdir)
    fig_4abc(m, outdir, n)
    fig_4abc(m, outdir, n, ln=True)
    fig_4de(m, outdir, n)
    fig_4de(m, outdir, n, ln=True)
    fig_4f(m, outdir, n)
    fig_S1(m, outdir)
    fig_S3(m, outdir, n_detail=n)
    fig_S4(m, outdir)
    fig_S5(m, outdir)
