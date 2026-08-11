"""
Every number in HIPO_model_theory.docx, checked against the code.

The theory document, the manuscript and all figures read the chain from the
T_d-rich terminus; the code propagates from the T_a-rich terminus. That mapping
is applied by ``core.manuscript_view`` and ``core.to_paper_index``.

This file exists because the document was, for a while, internally mixed: its
Tables 2-5 used the code convention while Table 10 and Section 8 used the
manuscript convention, so "[T_F]" meant different matrices in different tables.
Nothing caught it, because no test compared the document to the code. Now one
does.

    cd src && PYTHONPATH=. python ../tests/test_manuscript_convention.py
"""
import numpy as np
from hipo import core as H

m = H.MODEL
v = H.manuscript_view(m)
d = H.DPHI
NAMES = ["Ta", "Tb", "Tc", "Td"]
fails = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok:
        fails.append(name)
    print(f"  {name:<48} {got:>12.6g}  doc {want:<12g} {'PASS' if ok else '**FAIL**'}")


def limit_of(T):
    return NAMES[int(np.argmax(v.P_T @ np.linalg.matrix_power(T, 400)))]


print("=== the mapping itself ===")
assert np.array_equal(v.T_F, np.asarray(m.T_rev)), "[T_F] must be code T_rev"
assert np.array_equal(v.T_R, np.asarray(m.T_fwd)), "[T_R] must be code T_fwd"
print("  [T_F] == code T_rev, [T_R] == code T_fwd                       PASS")
for lab, T, want in (("[T_F] funnels to", v.T_F, "Ta"), ("[T_R] funnels to", v.T_R, "Td")):
    got = limit_of(T)
    ok = got == want
    if not ok:
        fails.append(lab)
    print(f"  {lab:<48} {got:>12}  doc {want:<12} {'PASS' if ok else '**FAIL**'}")

print("\n=== Table 2 / Table 3 — the transition matrices ===")
T2 = np.array([[0.99999980, 1.96e-7, 0, 0], [0, 0, 0.40507, 0.59493],
               [0.99999980, 1.96e-7, 0, 0], [0, 0, 0.40507, 0.59493]])
T3 = np.array([[0.73530, 0, 0.26470, 0], [0.73530, 0, 0.26470, 0],
               [0, 3.71e-7, 0, 0.99999963], [0, 3.71e-7, 0, 0.99999963]])
check("Table 2 [T_F] max deviation", float(np.abs(v.T_F - T2).max()), 0.0, 5e-6)
check("Table 3 [T_R] max deviation", float(np.abs(v.T_R - T3).max()), 0.0, 5e-6)

print("\n=== Table 1 — <cos^2>, and why the grid choice is safe ===")
print("  Reported on the raw 15-deg DFT scan grid (= spreadsheet AE28/U28).")
_c_scan = H.mean_cos2(m, "scan")
_c_grid = H.mean_cos2(m, "grid")
for _i, (_nm, _w) in enumerate(zip(NAMES,
        [0.9514813, 0.4615182, 0.8431211, 0.9492085])):
    check(f"<cos^2> {_nm} on the scan grid", float(_c_scan[_i]), _w, 1e-6)
print("  The 1-degree interpolated grid gives "
      f"{np.array2string(_c_grid, precision=4)} -- a ~1% difference, because")
print("  cos^2 varies inside each 15-deg interval.  Both are correct for what")
print("  they are; the scan-grid value is reported because <cos^2> is a dimer")
print("  descriptor that feeds nothing.  See core.mean_cos2.")

# The propagation layer must be indifferent to that choice.  P(T) is the integral
# of P alone, and the trapezoid rule is the exact integral of a linear
# interpolant, so the 1-degree grid must reproduce the tabulated P(T).
# m.P_phi_T is already row-normalised, so integrate the RAW weights instead.
_sp = np.asarray(m.scan_phi, dtype=np.float64)
_sw = np.asarray(m.scan_w, dtype=np.float64)
_lin = np.array([np.interp(np.asarray(H.PHI_GRID), _sp, _sw[i], period=360.0)
                 for i in range(4)])
_PT_grid = _lin.sum(1) * d
_PT_grid = _PT_grid / _PT_grid.sum()
check("P(T) from the 1-deg grid == tabulated P(T)",
      float(np.abs(_PT_grid - v.P_T).max()), 0.0, 1e-8)

print("\n=== Table 4 — second eigenvalues ===")
check("lambda_2 [T_F]", v.lam2_F, 0.594933, 1e-6)
check("lambda_2 [T_R]", v.lam2_R, 0.735304, 1e-6)

print("\n=== Table 5 — escape probabilities ===")
check("p_F  [T_F](Tc|Td)", v.p_F, 0.40507, 1e-5)
check("p_R  [T_R](Tc|Ta)", v.p_R, 0.26470, 1e-5)
check("[T_F](Tc|Ta) is zero", float(v.T_F[0, 2]), 0.0, 1e-15)
check("[T_R](Tc|Td) is zero", float(v.T_R[3, 2]), 0.0, 1e-15)

# escape_probabilities() is a separate code path from manuscript_view and must be
# pinned independently.  Mutation testing found that swapping its two return
# values left every test green -- and that labelling is precisely the thing that
# was reported crossed in four successive drafts.  Check the function, not just
# the view.
esc = H.escape_probabilities(m)
check("escape_probabilities()['p_F']", esc["p_F"], 0.40507, 1e-5)
check("escape_probabilities()['p_R']", esc["p_R"], 0.26470, 1e-5)
_swapped = abs(esc["p_F"] - 0.26470) < 1e-5 and abs(esc["p_R"] - 0.40507) < 1e-5
if _swapped:
    fails.append("escape_probabilities p_F/p_R swapped")
print(f"  {'p_F > p_R (Td basin leaks more readily)':<48} "
      f"{esc['p_F'] > esc['p_R']!s:>12}  doc True         "
      f"{'PASS' if esc['p_F'] > esc['p_R'] else '**FAIL**'}")
if not esc["p_F"] > esc["p_R"]:
    fails.append("p_F/p_R ordering")

print("\n=== Table 6 — divergence of [T_R]^s from the exact reverse of [T_F]^s ===")
DIV = {1: 0.000, 2: 0.129, 3: 0.259, 4: 0.368, 5: 0.454}
for s, want in DIV.items():
    Fs = np.linalg.matrix_power(v.T_F, s)
    Pn = v.P_T @ Fs
    exact = ((Fs * v.P_T[:, None]) / (Pn[None, :] + 1e-300)).T
    got = float(np.abs(exact - np.linalg.matrix_power(v.T_R, s)).max())
    check(f"s = {s}", got, want, 5e-4)

print("\n=== Section 4.4 — P(T) is not stationary under [T_F] ===")
check("max |P.[T_F] - P|", float(np.abs(v.P_T @ v.T_F - v.P_T).max()), 0.191, 1e-3)

print("\n=== Table 7 — T_c at k = 1, the Td/cis head ===")
T7 = {6: 0.1232, 10: 0.1034, 15: 0.0970, 20: 0.0957}
for n, want in T7.items():
    g = H.taut_profile(m, n)[:, 2]
    check(f"n = {n:<3} P(Tc) at k = 1", float(g[H.to_model_index(n, 1)]), want, 5e-4)

print("\n=== Table 9 — Phi_rest, phi_1 anchor, forward (matches Figure 4d) ===")
mt, mc = np.asarray(H.MASK_TRANS), np.asarray(H.MASK_CIS)
# Corrected after the anchor="last" fix (see core._accumulate_last).  The
# superseded values -- 5.2413 / 5.4790 / 5.8016 / 5.8635 / 5.8797 with P(cis)
# 0.5258 / 0.3643 / 0.2522 / 0.2112 / 0.1917 -- came from walking backwards with
# the opposite kernel and are the approximation, not the model.  Pinning them
# here pinned the error in place; the invariant that actually protects this axis
# now lives in test_marginalisation.py section 6b.
T9 = {3: (4.9060, 0.3959), 4: (5.2260, 0.5253), 5: (5.4449, 0.4050),
      10: (5.7809, 0.2960), 15: (5.8533, 0.2454), 20: (5.8731, 0.2191)}
for n, (ws, wc) in T9.items():
    J = np.asarray(H.anchor_phirest_joint(m, n, "last", "forward", True), dtype=np.float64)
    r = J.sum(0) * d
    r /= r.sum() * d
    q = np.maximum(r, 1e-300)
    check(f"n = {n:<3} S(Phi_rest)", float(-(r * np.log(q)).sum() * d), ws, 5e-4)
    check(f"n = {n:<3} P(cis)", float(r[mc].sum() * d), wc, 5e-4)

print("\n=== Table 10 — belief crossings, mixture rule ===")
T10 = {6: 1.885, 10: 3.381, 15: 5.234, 20: 7.083, 30: 10.815, 50: 18.227}
for n, want in T10.items():
    N = n - 1
    ks = np.arange(1, N + 1)
    diff = []
    for k in ks:
        a = v.P_T @ np.linalg.matrix_power(v.T_F, k - 1)
        b = v.P_T @ np.linalg.matrix_power(v.T_R, N - k)
        g = 0.5 * (a + b)
        g /= g.sum()
        diff.append(g[0] - g[3])
    diff = np.array(diff)
    s = np.flatnonzero(np.sign(diff[:-1]) != np.sign(diff[1:]))
    got = ks[s[0]] + diff[s[0]] / (diff[s[0]] - diff[s[0] + 1])
    check(f"n = {n:<3} crossing", float(got), want, 2e-3)

print("\n=== Section 8.1 — k* fractional position ===")
frac = np.log(v.lam2_R) / (np.log(v.lam2_F) + np.log(v.lam2_R))
check("k* fraction (doc primary)", float(frac), 0.3719, 5e-5)
check("k* fraction from the far end", float(1 - frac), 0.6281, 5e-5)

print("\n=== Table 11 — joint entropies under manuscript labels ===")
def S2(J):
    J = np.asarray(J, dtype=np.float64)
    J = J / (J.sum() * d * d)
    return float(-(J * np.log(np.maximum(J, 1e-300))).sum() * d * d)

T11 = {3: (9.6748, 9.7021, 9.7023), 4: (9.6429, 9.6515, 9.6898),
       5: (9.6100, 9.5686, 9.6648), 6: (9.5855, 9.4801, 9.6405),
       10: (9.5484, 9.2092, 9.5825), 15: (9.5430, 9.0650, 9.5617),
       20: (9.5426, 9.0206, 9.5572)}
for n, (wf, wr, wa) in T11.items():
    N = n - 1
    # manuscript forward == code reverse, and vice versa
    check(f"n = {n:<3} S_fwd", S2(H.pairwise_joint(m, 0, N - 1, "reverse", False)), wf, 5e-4)
    check(f"n = {n:<3} S_rev", S2(H.pairwise_joint(m, 0, N - 1, "forward", False)), wr, 5e-4)
    check(f"n = {n:<3} S_avg", S2(H.orientation_averaged_joint(m, 0, N - 1, False)), wa, 5e-4)

print("\n=== Section 9 — S_avg is monotonic and the JS excess saturates ===")
sa = [S2(H.orientation_averaged_joint(m, 0, n - 2, False)) for n in T11]
ok = bool(np.all(np.diff(sa) < 0))
if not ok:
    fails.append("S_avg monotonic")
print(f"  S_avg strictly decreasing in n                                 "
      f"{'PASS' if ok else '**FAIL**'}")
corr_len = 1.0 / abs(np.log(v.lam2_R))
check("1/|ln lambda_2([T_R])|, dimers", float(corr_len), 3.25, 5e-3)

print("\n" + "=" * 74)
if fails:
    print(f"{len(fails)} FAILURE(S): " + ", ".join(fails))
    raise SystemExit(1)
print("Theory document and code agree in every checked value.")
