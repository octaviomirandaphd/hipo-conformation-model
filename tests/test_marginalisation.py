"""
Marginalisation consistency — the cheapest test that catches axis-convention bugs.

A joint distribution and its marginals must describe the same ensemble.  Every
joint the model produces is checked against the corresponding marginal computed
by an independent code path.  Run:

    cd src && PYTHONPATH=. python ../tests/test_marginalisation.py

Rationale.  The orientation-averaged pairwise joint was for some time computed
as 0.5*(jf + jr) with no transpose on the reverse term.  Each individual joint
was correct and every direct numerical check passed, but the average did not
marginalise to eq 6 and its mutual information collapsed to ~1e-06.  This file
exists so that class of error cannot return silently.
"""
import numpy as np
from hipo import core as H

m, d = H.MODEL, H.DPHI
NM = [3, 4, 5, 6, 10, 15, 20]
TOL = 1e-12
fails = []


def n1(y):
    y = np.asarray(y, float)
    return y / (y.sum() * d)


def check(name, err, tol=TOL):
    ok = err < tol
    if not ok:
        fails.append(name)
    print(f"  {name:<58} {err:>10.3e}  {'PASS' if ok else '**FAIL**'}")


print("=== 1. inputs are well formed ===")
Tf, Tr, PT = np.asarray(m.T_fwd), np.asarray(m.T_rev), np.asarray(m.P_T)
check("T_fwd rows sum to 1", abs(Tf.sum(1) - 1).max())
check("T_rev rows sum to 1", abs(Tr.sum(1) - 1).max())
check("P(T) sums to 1", abs(PT.sum() - 1))
check("P(phi|T) rows integrate to 1", abs(np.asarray(m.P_phi_T).sum(1) * d - 1).max())
Pn = Tf.T @ PT
# NOTE: this identity is near-circular.  T_rev is COMPUTED from it, so the check
# reduces to whether (x/y)*y round-trips in float64 -- it does, giving exactly
# 0.0, which is a statement about IEEE arithmetic, not about the model.
check("one-step Bayes (circular, expect exact 0)",
      np.abs(Tf * PT[:, None] - (Tr * Pn[:, None]).T).max())

# The non-circular test: rebuild T_rev from the trimer populations directly, the
# way the parameter spreadsheet does, and compare.  Permitted successors are
#   reverse:  Ta,Tc -> {Ta,Tb}      Tb,Td -> {Tc,Td}
# with probabilities equal to the equilibrium populations renormalised over each
# set.  Agreement HERE is evidence; agreement above is not.
_tri = np.zeros((4, 4))
_tri[0] = _tri[2] = [PT[0] / (PT[0] + PT[1]), PT[1] / (PT[0] + PT[1]), 0, 0]
_tri[1] = _tri[3] = [0, 0, PT[2] / (PT[2] + PT[3]), PT[3] / (PT[2] + PT[3])]
_rel = np.where(_tri > 0, np.abs(Tr - _tri) / np.maximum(_tri, 1e-300), 0.0)
check("T_rev(Bayes) vs T_rev(trimer), relative", float(_rel.max()), 1e-15)

# The forward matrix against its own trimer expression.  Limited by the 8
# significant figures stored in data/tautomer_parameters.csv, not by arithmetic:
# 8e-10 absolute.  Immaterial at reported precision, but this -- not 1e-16 -- is
# the true precision floor of the whole parameter chain.
_trif = np.zeros((4, 4))
_trif[0] = _trif[1] = [PT[0] / (PT[0] + PT[2]), 0, PT[2] / (PT[0] + PT[2]), 0]
_trif[2] = _trif[3] = [0, PT[1] / (PT[1] + PT[3]), 0, PT[3] / (PT[1] + PT[3])]
check("T_fwd(csv) vs T_fwd(trimer), absolute", float(np.abs(Tf - _trif).max()), 1e-8)
print(f"  {'P(T) is NOT stationary under T_fwd (expected)':<58} "
      f"{np.abs(PT @ Tf - PT).max():>10.3e}  (informational)")

print("\n=== 2. single-direction pairwise joints vs their own marginals ===")
for n in NM:
    N = n - 1
    for dr in ("forward", "reverse"):
        J = np.asarray(H.pairwise_joint(m, 0, N - 1, dr, True), float)
        check(f"n={n:<3} {dr:<8} axis 0 -> marginal(0)",
              np.abs(n1(J.sum(1) * d) - np.asarray(H.marginal(m, 0, dr))).max())
        check(f"n={n:<3} {dr:<8} axis 1 -> marginal(N-1)",
              np.abs(n1(J.sum(0) * d) - np.asarray(H.marginal(m, N - 1, dr))).max())

print("\n=== 3. orientation-averaged joint vs eq 6  (THE REGRESSION TEST) ===")
for n in NM:
    N = n - 1
    J = np.asarray(H.orientation_averaged_joint(m, 0, N - 1, True), float)
    check(f"n={n:<3} axis 0 -> avg_marginal(k=0)",
          np.abs(n1(J.sum(1) * d) - np.asarray(H.avg_marginal(m, n, 0))).max())
    check(f"n={n:<3} axis 1 -> avg_marginal(k=N-1)",
          np.abs(n1(J.sum(0) * d) - np.asarray(H.avg_marginal(m, n, N - 1))).max())

print("\n=== 4. the un-transposed average FAILS the same test (guard) ===")
print("  If any line below reports a small number, the transpose has stopped")
print("  mattering and something else has changed.  Expect ~1e-03.")
for n in (5, 20):
    N = n - 1
    jf = np.asarray(H.pairwise_joint(m, 0, N - 1, "forward", False), float)
    jr = np.asarray(H.pairwise_joint(m, 0, N - 1, "reverse", False), float)
    A = 0.5 * (jf + jr); A /= A.sum() * d * d
    e = np.abs(n1(A.sum(1) * d) - np.asarray(H.avg_marginal(m, n, 0))).max()
    ok = e > 1e-6
    if not ok:
        fails.append(f"guard n={n}")
    print(f"  n={n:<3} 0.5*(jf+jr) axis 0 vs eq 6 {e:>10.3e}  "
          f"{'PASS (correctly inconsistent)' if ok else '**FAIL — guard broke**'}")

print("\n=== 5. mutual information survives the average ===")
def MI(z):
    z = np.asarray(z, float); z = z / (z.sum() * d * d)
    a, b = z.sum(1) * d, z.sum(0) * d
    o = np.outer(a, b); k = z > 1e-300
    return float((z[k] * np.log(z[k] / np.maximum(o[k], 1e-300))).sum() * d * d)

print(f"  {'n':>4} {'I(fwd)':>12} {'I(rev)':>12} {'I(avg)':>12}")
for n in NM:
    N = n - 1
    ia = MI(H.orientation_averaged_joint(m, 0, N - 1, False))
    print(f"  {n:>4} {MI(H.pairwise_joint(m,0,N-1,'forward',False)):>12.4e} "
          f"{MI(H.pairwise_joint(m,0,N-1,'reverse',False)):>12.4e} {ia:>12.4e}")
    if ia < 1e-3:
        fails.append(f"I(avg) collapsed at n={n}")
print("  I(avg) must stay >= 1e-3 at every n; collapse indicates a lost transpose.")

print("\n=== 6. anchor / Phi_rest joints ===")
for n in NM:
    N = n - 1
    for anc in ("first", "last"):
        J = np.asarray(H.anchor_phirest_joint(m, n, anc, "forward", True), float)
        ref = (np.asarray(H.marginal(m, 0, "forward")) if anc == "first"
               else np.asarray(H.marginal(m, N - 1, "forward")))
        check(f"n={n:<3} anchor={anc:<6} anchor axis -> marginal",
              np.abs(n1(J.sum(1) * d) - ref).max())
        check(f"n={n:<3} anchor={anc:<6} integrates to 1", abs(J.sum() * d * d - 1))

print("\n=== 6b. total-twist invariant across the two anchors  [KEY REGRESSION] ===")
print("  Both anchors describe the SAME chain, so convolving each joint's two")
print("  axes must give the same total twist phi_1 + ... + phi_N.  This needs no")
print("  second implementation and no golden number, and it is what finally")
print("  caught anchor='last' being built by walking backwards with the opposite")
print("  kernel (residual 2e-04 to 8e-04 for n >= 4, exact only at n = 3).")


def _total_twist(J):
    J = np.asarray(J, dtype=np.float64)
    tot = np.zeros(J.shape[0])
    for i in range(J.shape[0]):
        tot += np.roll(J[i], i) * d      # shift Phi_rest by the anchor angle
    tot *= d
    return tot / (tot.sum() * d)


for n in NM:
    f = _total_twist(H.anchor_phirest_joint(m, n, "first", "forward", True))
    l = _total_twist(H.anchor_phirest_joint(m, n, "last", "forward", True))
    check(f"n={n:<3} total twist: first anchor == last anchor",
          float(np.abs(f - l).max()), 1e-14)

print("\n=== 6c. structural: anchor='last' against brute-force enumeration ===")
print("  4^N tautomer paths summed explicitly at n = 4; no shared code with core.")
import itertools
_Pp = np.asarray(m.P_phi_T, float)
_PT = np.asarray(m.P_T, float)
_Tf = np.asarray(m.T_fwd, float)
_M = H.N_GRID
for _n in (3, 4):
    _N = _n - 1
    _J = np.zeros((_M, _M))
    for path in itertools.product(range(4), repeat=_N):
        w = _PT[path[0]]
        for a, b in zip(path, path[1:]):
            w *= _Tf[a, b]
        if w < 1e-30:
            continue
        acc = _Pp[path[0]].copy()
        for kk in range(1, _N - 1):
            acc = np.real(np.fft.ifft(np.fft.fft(acc) * np.fft.fft(_Pp[path[kk]]))) * d
        _J += w * np.outer(_Pp[path[_N - 1]], acc)
    _J = np.maximum(_J, 0.0); _J /= _J.sum() * d * d
    _C = np.asarray(H.anchor_phirest_joint(m, _n, "last", "forward", True), float)
    check(f"n={_n:<3} anchor='last' == brute force", float(np.abs(_C - _J).max()), 1e-15)

print("\n=== 7. structural: at n=3 Phi_rest is a single angle ===")
J = np.asarray(H.anchor_phirest_joint(m, 3, "first", "forward", True), float)
check("n=3 Phi_rest axis == P(phi_2) exactly (one transition, one angle)",
      np.abs(n1(J.sum(0) * d) - np.asarray(H.marginal(m, 1, "forward"))).max())

print("\n=== 8. avg_marginal normalisation along the whole chain ===")
for n in NM:
    e = max(abs(float(np.asarray(H.avg_marginal(m, n, k)).sum() * d) - 1)
            for k in range(n - 1))
    check(f"n={n:<3} every avg_marginal integrates to 1", e)

print("\n=== 8b. figure-facing helpers  (mutants g, f2, l) ===")
print("  These three were exercised by no test at all, yet each one places")
print("  something on a published figure.  Asserting on the plotted values, not")
print("  on a reimplementation of the formula.")

# --- k_star: drives the k* marker on Figures 4f and S4 -------------------
# The convention test re-derived the fraction inline, so k_star() itself had no
# caller in tests/ and could return anything.  Call the function here.
_ks20, _fr = H.k_star(m, 20)
check("k_star(m,20) model coordinate", abs(_ks20 - 12.30595), 5e-5)
check("k_star fraction", abs(_fr - 0.6281085), 5e-7)
_ks_paper = (20 - 1) - _ks20 + 1
check("k* as printed on Fig 4f (paper coords)", abs(_ks_paper - 7.69405), 5e-5)
for _n in (6, 10, 30, 50):
    _k, _f = H.k_star(m, _n)
    check(f"n={_n:<3} k* fraction chain-length independent", abs(_f - _fr), 1e-12)

# --- to_paper_index / to_model_index: the chain-axis mapping -------------
check("to_paper_index(20, 0) == 19", abs(H.to_paper_index(20, 0) - 19), 0.5)
check("to_paper_index(20, 18) == 1", abs(H.to_paper_index(20, 18) - 1), 0.5)
_rt = max(abs(H.to_model_index(20, H.to_paper_index(20, k)) - k) for k in range(19))
check("to_model_index inverts to_paper_index", float(_rt), 0.5)

# --- the plotted chain axis itself (mutant f2) ---------------------------
# Reversing _paper_positions mirrors Figures 4f and S4 -- the Td and Ta ends
# swap and k* moves to the mirror position -- and no test noticed, because the
# only figure test compares the two COPIES of the code to each other.  Assert on
# the plotted arrays instead.
from hipo import _figblock as _FB
for _n in (6, 20):
    _N = _n - 1
    _x = np.asarray(_FB._paper_positions(_N))
    check(f"n={_n:<3} chain axis starts at paper k=N", abs(_x[0] - _N), 0.5)
    check(f"n={_n:<3} chain axis ends at paper k=1", abs(_x[-1] - 1), 0.5)
    _g = H.taut_profile(m, _n)
    # model index 0 is the Ta-rich end and must be plotted at paper k = N
    _td_at = _x[int(np.argmax(_g[:, 3]))]
    _ta_at = _x[int(np.argmax(_g[:, 0]))]
    ok = (_td_at == 1) and (_ta_at == _N)
    if not ok:
        fails.append(f"chain axis orientation n={_n}")
    print(f"  {'n=' + str(_n) + ' Td peak at paper k=1, Ta peak at paper k=N':<58} "
          f"{'Td@' + str(_td_at) + ' Ta@' + str(_ta_at):>10}  "
          f"{'PASS' if ok else '**FAIL**'}")

print("\n=== 9. closed form  gamma_1(Tc) = 1/2 P(Tc) ===")
g = 0.5 * (float(np.asarray(H.belief_at(m, 19, "forward"))[2])
           + float(np.asarray(H.belief_at(m, 0, "reverse"))[2]))
print(f"  computed {g:.4f}   closed form {0.5*float(PT[2]):.4f}   "
      f"diff {abs(g-0.5*float(PT[2])):.4f}")

print("\n" + "=" * 70)
if fails:
    print(f"{len(fails)} FAILURE(S): " + ", ".join(fails))
    raise SystemExit(1)
print("All marginalisation checks passed.")
