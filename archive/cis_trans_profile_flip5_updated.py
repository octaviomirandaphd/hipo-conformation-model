"""
Cis/Trans conformation profile along IPD polymer chains
========================================================
For each n-mer chain length, computes and plots:

  P(trans, k) = P(phi_k within +-30 of 0 deg)   -- trans-like
  P(cis,   k) = P(phi_k within +-30 of 180 deg) -- cis-like
  P(other, k) = 1 - P(trans,k) - P(cis,k)       -- intermediate

at each dimer position k = 1,...,N using the orientation-averaged
marginal P_avg(phi_k) = 0.5 * [P_fwd(phi_k) + P_rev(phi_k)].

Also computes:
  Chain-averaged <P(cis)>  and <P(trans)> vs N
  showing convergence to limiting values as chain grows.

Physical interpretation:
  Forward (head->tail) funnels toward Td (cis, phi~180)
  Reverse (tail->head) funnels toward Ta (trans, phi~0)
  => gradient from trans-rich at head to cis-rich at tail
  => cis dominates overall because Td's preference is sharper
     than Ta's (Td is essentially all-cis; Ta is mostly-trans
     but with a broader distribution)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ── rcParams matching polymer_conformation8.py ────────────────────────────────
BASE_FS = 10
matplotlib.rcParams.update({
    "font.size":        BASE_FS * 1.5,
    "axes.titlesize":   BASE_FS * 1.5,
    "axes.labelsize":   BASE_FS * 1.5,
    "xtick.labelsize":  BASE_FS * 1.5,
    "ytick.labelsize":  BASE_FS * 1.5,
    "legend.fontsize":  BASE_FS * 1.5,
    "figure.titlesize": BASE_FS * 1.8,
})

# ── Angle grid ────────────────────────────────────────────────────────────────
N_GRID   = 360
PHI_GRID = np.linspace(-180.0, 180.0, N_GRID, endpoint=False)
DPH      = PHI_GRID[1] - PHI_GRID[0]

# ── P(phi|T) data (validated) ─────────────────────────────────────────────────
phi_raw = np.array([
    -180,-165,-150,-135,-120,-105,-90,-75,-60,-45,-30,-15,
       0,  15,  30,  45,  60,  75, 90,105,120,135,150,165,180
], dtype=float)

Pphi_raw = {
    "Ta": np.array([0.337991667,0.162442177,0.023582201,0.00077091,9.85523e-7,
        2.52179e-8,8.20422e-9,5.80932e-8,3.36813e-6,0.0026263,0.067802355,
        0.514903052,1.0,0.514903052,0.067802355,0.0026263,3.36813e-6,
        5.80932e-8,8.20422e-9,2.52179e-8,9.85523e-7,0.00077091,0.023582201,
        0.162442177,0.337991667]),
    "Tb": np.array([1.46247e-12,5.69999e-10,1.3249e-8,5.83401e-8,5.25821e-8,
        1.32286e-9,5.11025e-10,7.62978e-10,5.05591e-8,5.64536e-8,4.67436e-8,
        1.48336e-9,6.76823e-10,1.48336e-9,4.67436e-8,5.64536e-8,5.05591e-8,
        7.62978e-10,5.11025e-10,1.32286e-9,5.25821e-8,5.83401e-8,1.3249e-8,
        5.69999e-10,1.46247e-12]),
    "Tc": np.array([5.51567e-10,1.07225e-7,2.7101e-6,7.57204e-6,8.67275e-6,
        1.22552e-7,1.50063e-7,1.7069e-6,0.002719824,0.025588394,0.189640966,
        0.28555259,0.030513107,0.28555259,0.189640966,0.025588394,
        0.002719824,1.7069e-6,1.50063e-7,1.22552e-7,8.67275e-6,7.57204e-6,
        2.7101e-6,1.07225e-7,5.51567e-10]),
    "Td": np.array([0.70325558,0.35188851,0.056378998,0.002019016,3.13868e-5,
        3.24915e-8,6.96081e-9,1.01101e-8,3.90524e-7,3.28538e-7,8.71962e-8,
        5.83736e-9,4.63302e-11,5.83736e-9,8.71962e-8,3.28538e-7,3.90524e-7,
        1.01101e-8,6.96081e-9,3.24915e-8,3.13868e-5,0.002019016,0.056378998,
        0.35188851,0.70325558]),
}

TAUT_NAMES = ["Ta", "Tb", "Tc", "Td"]
P_phi_T = np.zeros((4, N_GRID))
for i, name in enumerate(TAUT_NAMES):
    interp = np.interp(PHI_GRID, phi_raw, Pphi_raw[name], period=360.0)
    interp = np.maximum(interp, 0.0)
    P_phi_T[i] = interp / (interp.sum() * DPH)

# ── Validated tautomer populations and transition matrices ────────────────────
P_T = np.array([0.52946541, 1.03943e-7, 0.19059783, 0.279936655])
P_T /= P_T.sum()

T_fwd_raw = np.array([
    [0.73530404, 0.0,        0.26469596, 0.0        ],
    [0.73530404, 0.0,        0.26469596, 0.0        ],
    [0.0,        3.71308e-7, 0.0,        0.999999629],
    [0.0,        3.71308e-7, 0.0,        0.999999629],
])
T_fwd = T_fwd_raw / T_fwd_raw.sum(axis=1, keepdims=True)
P_Tj  = T_fwd.T @ P_T
T_rev = ((T_fwd * P_T[:, None]) / (P_Tj[None, :] + 1e-300)).T

# ── Cis/trans masks (+-30 deg windows) ───────────────────────────────────────
mask_trans = (PHI_GRID >= -30) & (PHI_GRID <= 30)
mask_cis   = (PHI_GRID >= 150) | (PHI_GRID <= -150)

def prob_in_mask(p_norm, mask):
    return float(p_norm[mask].sum() * DPH)

# ── Belief propagation — orientation-averaged marginal at dimer k ─────────────
def avg_marginal(n_monomers, k):
    """
    Orientation-averaged normalised marginal P(phi) at dimer position k
    (0-based) in an n-mer chain.
    """
    def propagate(msg, T_mat):
        return msg @ T_mat

    # Forward
    msg_f = P_T.copy()
    for _ in range(k):
        msg_f = propagate(msg_f, T_fwd)
    mf = msg_f @ P_phi_T
    mf = np.maximum(mf, 0.0)
    mf /= mf.sum() * DPH

    # Reverse (start from tail end: propagate N-1-k steps in reverse)
    N = n_monomers - 1
    msg_r = P_T.copy()
    for _ in range(N - 1 - k):
        msg_r = propagate(msg_r, T_rev)
    mr = msg_r @ P_phi_T
    mr = np.maximum(mr, 0.0)
    mr /= mr.sum() * DPH

    m = 0.5 * (mf + mr)
    return m / (m.sum() * DPH)

# ── Chain lengths ─────────────────────────────────────────────────────────────
ALL_NMERS    = [3, 4, 5, 6, 10, 15]   # 6-panel figure
STANDALONE_N = 20                      # separate single-panel plot

# ── Compute cis/trans profiles for all n-mers ─────────────────────────────────
print("=" * 60)
print("CIS/TRANS PROFILE vs DIMER POSITION k")
print("=" * 60)

profiles = {}
for n in ALL_NMERS + [STANDALONE_N]:
    N = n - 1
    p_trans = []
    p_cis   = []
    for k in range(N):
        m = avg_marginal(n, k)
        pt = prob_in_mask(m, mask_trans)
        pc = prob_in_mask(m, mask_cis)
        p_trans.append(pt)
        p_cis.append(pc)
    profiles[n] = {
        "trans": np.array(p_trans),
        "cis":   np.array(p_cis),
    }
    avg_t = np.mean(p_trans)
    avg_c = np.mean(p_cis)
    print(f"\n  n={n} (N={N} dimers):")
    print(f"    k=1 (head): trans={p_trans[0]:.3f}  cis={p_cis[0]:.3f}")
    if N > 1:
        mid = N // 2
        print(f"    k={mid+1} (mid):  trans={p_trans[mid]:.3f}  cis={p_cis[mid]:.3f}")
    print(f"    k={N} (tail): trans={p_trans[-1]:.3f}  cis={p_cis[-1]:.3f}")
    print(f"    Chain avg:  <trans>={avg_t:.3f}  <cis>={avg_c:.3f}")

# ── Chain-averaged values vs N (convergence) ──────────────────────────────────
# Extend to larger N to show convergence clearly
N_range = list(range(2, 51))
avg_trans_vs_N = []
avg_cis_vs_N   = []
avg_other_vs_N = []

for n in N_range:
    N = n - 1
    pt_list = []
    pc_list = []
    for k in range(N):
        m = avg_marginal(n, k)
        pt_list.append(prob_in_mask(m, mask_trans))
        pc_list.append(prob_in_mask(m, mask_cis))
    avg_trans_vs_N.append(np.mean(pt_list))
    avg_cis_vs_N.append(np.mean(pc_list))
    avg_other_vs_N.append(1.0 - np.mean(pt_list) - np.mean(pc_list))

print("\n" + "=" * 60)
print("CHAIN-AVERAGED CIS/TRANS vs N (convergence)")
print("=" * 60)
print(f"  {'N':>4}  {'<trans>':>8}  {'<cis>':>8}  {'<other>':>8}")
for i, n in enumerate(N_range[::5]):
    idx = N_range.index(n)
    print(f"  {n:>4}  {avg_trans_vs_N[idx]:>8.4f}  "
          f"{avg_cis_vs_N[idx]:>8.4f}  {avg_other_vs_N[idx]:>8.4f}")

# ── Chain-end Ta/Td tautomer belief vs N ─────────────────────────────────
# P(Ta) at head (k=1) and P(Td) at tail (k=N) vs chain length
ta_head_vs_N = []
td_tail_vs_N = []
for n in N_range:
    N = n - 1
    # Head dimer (k=0): 0 forward steps, N-1 reverse steps
    mf = P_T.copy()
    mr = P_T.copy()
    for _ in range(N - 1): mr = mr @ T_rev
    m_head = 0.5*(mf + mr); m_head /= m_head.sum()
    ta_head_vs_N.append(float(m_head[0]))
    # Tail dimer (k=N-1): N-1 forward steps, 0 reverse steps
    mf2 = P_T.copy()
    for _ in range(N - 1): mf2 = mf2 @ T_fwd
    mr2 = P_T.copy()
    m_tail = 0.5*(mf2 + mr2); m_tail /= m_tail.sum()
    td_tail_vs_N.append(float(m_tail[3]))



# ── Compute tautomer belief profiles for all n-mers ───────────────────────────
taut_profiles = {}
for n in ALL_NMERS + [STANDALONE_N]:
    N = n - 1
    ta_p = []; tc_p = []; td_p = []
    for k in range(N):
        msg_f = P_T.copy()
        for _ in range(k): msg_f = msg_f @ T_fwd
        msg_r = P_T.copy()
        for _ in range(N - 1 - k): msg_r = msg_r @ T_rev
        msg_avg = 0.5 * (msg_f + msg_r); msg_avg /= msg_avg.sum()
        ta_p.append(float(msg_avg[0]))
        tc_p.append(float(msg_avg[2]))
        td_p.append(float(msg_avg[3]))
    taut_profiles[n] = {
        "ta": np.array(ta_p),
        "tc": np.array(tc_p),
        "td": np.array(td_p),
    }

C_TA = "#D32F2F"; C_TC = "#1565C0"; C_TD = "#2E7D32"

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Cis/Trans profile per n-mer (6-panel) + standalone 20-mer
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.lines as mlines

def draw_panel(ax, n, show_legend=False):
    """Draw one cis/trans profile panel onto ax for chain length n."""
    N         = n - 1
    k_pos     = np.arange(1, N + 1)
    k_pos_rev = k_pos[::-1]          # reversed: Td/cis on left, Ta/trans on right
    prof      = profiles[n]
    tp        = taut_profiles[n]
    kstar      = 0.629 * (N - 1) + 1
    kstar_plot = N - kstar + 1       # position in reversed data space

    # Tautomer belief — solid lines with filled markers
    ax.plot(k_pos_rev, tp["ta"], color=C_TA, linewidth=1.8, marker="D", markersize=6)
    ax.plot(k_pos_rev, tp["tc"], color=C_TC, linewidth=1.8, marker="^", markersize=6)
    ax.plot(k_pos_rev, tp["td"], color=C_TD, linewidth=1.8, marker="v", markersize=6)
    # Cis/trans — dashed lines with open markers
    ax.plot(k_pos_rev, prof["trans"], color=C_TA, linewidth=1.8, linestyle="--",
            marker="D", markersize=6, markerfacecolor="none", markeredgewidth=1.2)
    ax.plot(k_pos_rev, prof["cis"],   color=C_TD, linewidth=1.8, linestyle="--",
            marker="v", markersize=6, markerfacecolor="none", markeredgewidth=1.2)

    # Conflict zone shading — only for n >= 10
    ta_arr = tp["ta"]; td_arr = tp["td"]
    in_conflict = np.abs(ta_arr - td_arr) < 0.10
    if n >= 10 and in_conflict.any():
        cz_xs  = k_pos_rev[in_conflict]
        cz_min = cz_xs.min() - 0.5
        cz_max = cz_xs.max() + 0.5
        ax.axvspan(cz_min, cz_max, alpha=0.12, color=C_TC, zorder=0)
        ax.annotate("Tc bridge\nregion",
                    xy=(cz_min + 0.3, 0.93), xycoords=("data", "axes fraction"),
                    fontsize=BASE_FS*1.0, color=C_TC, ha="left", va="top")

    # k* dashed line + label
    ax.axvline(kstar_plot, color=C_TC, linewidth=1.2, linestyle="--", alpha=0.85)
    ax.annotate(f"k*={kstar_plot:.1f}",
                xy=(kstar_plot, 0.20), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points",
                fontsize=BASE_FS*1.1, color=C_TC, va="bottom", ha="left")

    ax.axhline(0, color="k", linewidth=0.4)
    ax.set_title(f"{n}-mer  (N={N} dimers)", pad=20)
    ax.set_xlabel("Dimer position k")
    ax.set_ylabel("Probability")
    ax.set_xlim(0.5, N + 0.5)
    ax.set_xticks(k_pos if N <= 10 else k_pos[::2])
    ax.set_ylim(0, 1)

    ax.annotate("tail\n(→Td)", xy=(0, 0), xytext=(0.06, 0.62),
                textcoords=("axes fraction", "axes fraction"),
                fontsize=BASE_FS, ha="left", color="dimgray", alpha=0.8)
    ax.annotate("head\n(Ta→)", xy=(0, 0), xytext=(0.94, 0.62),
                textcoords=("axes fraction", "axes fraction"),
                fontsize=BASE_FS, ha="right", color="dimgray", alpha=0.8)

    if show_legend:
        h_ta    = mlines.Line2D([],[],color=C_TA,lw=1.8,marker="D",ms=6,label="Ta")
        h_trans = mlines.Line2D([],[],color=C_TA,lw=1.8,ls="--",marker="D",ms=6,
                                mfc="none",mew=1.2,label="trans")
        h_tc    = mlines.Line2D([],[],color=C_TC,lw=1.8,marker="^",ms=6,label="Tc")
        h_kstar = mlines.Line2D([],[],color=C_TC,lw=1.2,ls="--",label="k* (Tc bridge)")
        h_td    = mlines.Line2D([],[],color=C_TD,lw=1.8,marker="v",ms=6,label="Td")
        h_cis   = mlines.Line2D([],[],color=C_TD,lw=1.8,ls="--",marker="v",ms=6,
                                mfc="none",mew=1.2,label="cis")
        ax.legend(handles=[h_ta, h_tc, h_td, h_trans, h_kstar, h_cis],
                  loc="upper right", ncol=2, fontsize=BASE_FS*0.95)

SUPTITLE = (
    "Cis/Trans Conformation Profile Along Chain\n"
    "P(trans) = P(φₖ within ±30° of 0°)   "
    "P(cis) = P(φₖ within ±30° of 180°)"
)

# 6-panel figure
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 9))
fig1.suptitle(SUPTITLE, fontweight="bold")
for ax, n in zip(axes1.flatten(), ALL_NMERS):
    draw_panel(ax, n, show_legend=(n == ALL_NMERS[0]))
fig1.tight_layout()
fig1.savefig("plot_cis_trans_profile.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\nSaved: plot_cis_trans_profile.png")

# Standalone 20-mer
fig_s, ax_s = plt.subplots(figsize=(9, 6))
fig_s.suptitle(SUPTITLE, fontweight="bold")
draw_panel(ax_s, STANDALONE_N, show_legend=True)
fig_s.tight_layout()
fig_s.savefig(f"plot_cis_trans_profile_{STANDALONE_N}mer.png", dpi=150, bbox_inches="tight")
plt.close(fig_s)
print(f"Saved: plot_cis_trans_profile_{STANDALONE_N}mer.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — All n-mers overlaid on same axes
# Normalise x-axis to [0,1] (fractional chain position) so all lengths compare
# ═══════════════════════════════════════════════════════════════════════════════
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(
    "Cis/Trans Profile — All Chain Lengths Overlaid\n"
    "x-axis normalised to fractional chain position (0=tail, 1=head)",
    fontweight="bold"
)

cmap_n = cm.plasma
norm_n = mcolors.Normalize(vmin=min(ALL_NMERS), vmax=max(ALL_NMERS))

for n in ALL_NMERS:
    N = n - 1
    prof = profiles[n]
    frac = np.linspace(1, 0, N)   # reversed: 1=tail/Td on left, 0=head/Ta on right
    c = cmap_n(norm_n(n))
    ax2a.plot(frac, prof["trans"], color=c, linewidth=2.0, label=f"n={n}")
    ax2b.plot(frac, prof["cis"],   color=c, linewidth=2.0, label=f"n={n}")

for ax, ylabel, title in [
    (ax2a, "P(trans, k)", "Trans probability vs fractional position"),
    (ax2b, "P(cis, k)",   "Cis probability vs fractional position"),
]:
    ax.axhline(0.5, color="k", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.axhline(0,   color="k", linewidth=0.4)
    ax.set_xlabel("Fractional chain position  (0=tail, 1=head)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=2)

fig2.tight_layout()
fig2.savefig("plot_cis_trans_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved: plot_cis_trans_overlay.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Two-panel convergence: cis/trans (left) + Ta/Td chain ends (right)
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.lines as mlines
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle(
    "Convergence of Chain-Averaged Cis/Trans Character and Chain-End Tautomer Belief vs Chain Length",
    fontweight="bold"
)

marked_n = ALL_NMERS + [30, 40, 50]

# ── Left panel: cis/trans chain average ──────────────────────────────────────
ax3a.plot(N_range, avg_trans_vs_N, color=C_TA, linewidth=2.2)
ax3a.plot(N_range, avg_cis_vs_N,   color=C_TD, linewidth=2.2)
for n in marked_n:
    if n in N_range:
        idx = N_range.index(n)
        ax3a.scatter([n], [avg_trans_vs_N[idx]], facecolors="none", edgecolors=C_TA,
                     s=80, zorder=5, marker="D", linewidth=1.4)
        ax3a.scatter([n], [avg_cis_vs_N[idx]],   facecolors="none", edgecolors=C_TD,
                     s=80, zorder=5, marker="v", linewidth=1.4)
lim_t = avg_trans_vs_N[-1]; lim_c = avg_cis_vs_N[-1]
ax3a.axhline(lim_t, color=C_TA, linewidth=0.8, linestyle=":", alpha=0.6)
ax3a.axhline(lim_c, color=C_TD, linewidth=0.8, linestyle=":", alpha=0.6)
ax3a.axhline(0, color="k", linewidth=0.4)
ax3a.set_xlabel("Chain length n (number of monomers)")
ax3a.set_ylabel("Chain-averaged probability")
ax3a.set_title("Chain-Averaged Cis/Trans Character", pad=20)
ax3a.set_xlim(2, 50); ax3a.set_ylim(0, 0.7)
h_trans_a = mlines.Line2D([],[],color=C_TA,lw=2.2,marker="D",ms=8,
                           markerfacecolor="none", markeredgewidth=1.4,
                           label="⟨P(trans)⟩ — chain average")
h_cis_a   = mlines.Line2D([],[],color=C_TD,lw=2.2,marker="v",ms=8,
                           markerfacecolor="none", markeredgewidth=1.4,
                           label="⟨P(cis)⟩ — chain average")
ax3a.legend(handles=[h_trans_a, h_cis_a], ncol=1, loc="lower right")

# ── Right panel: chain-end Ta/Td tautomer belief ─────────────────────────────
ax3b.plot(N_range, ta_head_vs_N, color=C_TA, linewidth=2.2)
ax3b.plot(N_range, td_tail_vs_N, color=C_TD, linewidth=2.2)
for n in marked_n:
    if n in N_range:
        idx = N_range.index(n)
        ax3b.scatter([n], [ta_head_vs_N[idx]], color=C_TA,
                     s=80, zorder=5, marker="D", edgecolors="white", linewidth=0.8)
        ax3b.scatter([n], [td_tail_vs_N[idx]], color=C_TD,
                     s=80, zorder=5, marker="v", edgecolors="white", linewidth=0.8)
lim_ta = ta_head_vs_N[-1]; lim_td = td_tail_vs_N[-1]
ax3b.axhline(lim_ta, color=C_TA, linewidth=0.8, linestyle=":", alpha=0.6)
ax3b.axhline(lim_td, color=C_TD, linewidth=0.8, linestyle=":", alpha=0.6)
ax3b.axhline(0, color="k", linewidth=0.4)
ax3b.set_xlabel("Chain length n (number of monomers)")
ax3b.set_ylabel("Chain-end tautomer belief probability")
ax3b.set_title("Chain-End Tautomer Belief: P(Ta) at head, P(Td) at tail", pad=20)
ax3b.set_xlim(2, 50); ax3b.set_ylim(0, 1)
h_ta_b = mlines.Line2D([],[],color=C_TA,lw=2.2,marker="D",ms=8,
                        label="P(Ta) at head end")
h_td_b = mlines.Line2D([],[],color=C_TD,lw=2.2,marker="v",ms=8,
                        label="P(Td) at Td tail end")
ax3b.legend(handles=[h_ta_b, h_td_b], ncol=1, loc="lower right")

fig3.tight_layout()
fig3.savefig("plot_cis_trans_convergence.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved: plot_cis_trans_convergence.png")
print("\nDone.")
