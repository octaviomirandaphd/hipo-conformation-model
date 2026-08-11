import numpy as np
from hipo import core as H
m=H.MODEL; Pp=np.asarray(m.P_phi_T); P_T=np.asarray(m.P_T)
Tf=np.asarray(m.T_fwd); Tr=np.asarray(m.T_rev); dφ=H.DPHI
# ---- adopt the ADDENDUM convention: [T_F] funnels to Ta  => addendum T_F = code T_rev
TF, TR = Tr, Tf
def a(k):   return P_T@np.linalg.matrix_power(TF,k-1)
def b(k,N): return P_T@np.linalg.matrix_power(TR,N-k)
def gam(k,N,rule):
    A,B=a(k),b(k,N); g=0.5*(A+B) if rule=="mix" else A*B/P_T; return g/g.sum()
def cross(n,rule,i=0,j=3):
    N=n-1; ks=np.arange(1,N+1)
    d=np.array([gam(k,N,rule)[i]-gam(k,N,rule)[j] for k in ks])
    s=np.where(np.sign(d[:-1])!=np.sign(d[1:]))[0]
    return np.nan if len(s)==0 else ks[s[0]]+d[s[0]]/(d[s[0]]-d[s[0]+1])

print("=== 5b. BELIEF CROSSINGS under the addendum convention (T_F funnels to Ta) ===")
cl={6:(1.885,0.221,2.354),10:(3.381,0.298,3.906),15:(5.234,0.326,5.805),
    20:(7.083,0.338,7.680),30:(10.815,0.351,11.414),50:(18.227,0.359,18.834)}
print(f"  {'n':>4} {'mixture':>9} {'claim':>8} {'(k-1)/(N-1)':>12} {'claim':>7} {'product':>9} {'claim':>8}")
for n in [6,10,15,20,30,50]:
    N=n-1; cm=cross(n,"mix"); cp=cross(n,"prod")
    print(f"  {n:>4} {cm:>9.3f} {cl[n][0]:>8.3f} {(cm-1)/(N-1):>12.3f} {cl[n][1]:>7.3f} {cp:>9.3f} {cl[n][2]:>8.3f}")

print("\n=== 6b. Tc populations, addendum convention ===")
clTc={6:(0.123,0.085,0.100,0.046),10:(0.103,0.040,0.063,0.015),
      15:(0.097,0.015,0.042,0.004),20:(0.096,0.006,0.031,0.001)}
print(f"  {'n':>4} {'term(max)':>10} {'int.min':>9} {'mean mix':>9} {'mean prod':>10}   claimed")
for n in [6,10,15,20]:
    N=n-1
    gm=np.array([gam(k,N,"mix")[2] for k in range(1,N+1)])
    gp=np.array([gam(k,N,"prod")[2] for k in range(1,N+1)])
    print(f"  {n:>4} {max(gm[0],gm[-1]):>10.4f} {gm[1:-1].min():>9.4f} {gm.mean():>9.4f} {gp.mean():>10.4f}   {clTc[n]}")
    if n==20: print(f"        gamma_1={gm[0]:.4f}  gamma_N={gm[-1]:.4f}   (doc: both ~ 0.0953/0.0957)")

print("\n=== 7. CIS/TRANS CHAIN AVERAGES (cis_trans_interpretation §3) ===")
cl2={3:(0.534,0.396),5:(0.485,0.471),10:(0.444,0.511),20:(0.405,0.555),50:(0.385,0.578)}
print(f"  {'n':>4} {'<trans>':>9} {'claim':>7} {'<cis>':>8} {'claim':>7} {'<other>':>8}")
for n in [3,5,10,20,50]:
    tr,ci=H.cis_trans_profile(m,n)
    print(f"  {n:>4} {tr.mean():>9.3f} {cl2[n][0]:>7.3f} {ci.mean():>8.3f} {cl2[n][1]:>7.3f} {1-tr.mean()-ci.mean():>8.3f}")
tr,ci=H.cis_trans_profile(m,201)
print(f"   n=201 (~inf) <trans>={tr.mean():.3f} (claim 0.380)  <cis>={ci.mean():.3f} (claim 0.583)")

print("\n=== 8. JOINT ENTROPY S(phi_1,phi_N) (entropy_correction_addendum_v2) ===")
def ent2(J):
    J=np.asarray(J); J=J/(J.sum()*dφ**2); p=np.maximum(J,1e-300)
    return -float((p*np.log(p)).sum()*dφ**2)
# S_avg claims are the CORRECTED values (reverse joint transposed before
# averaging).  The superseded values -- 9.7319 / 9.7346 / 9.7247 / 9.7121 /
# 9.6755 / 9.6605 / 9.6571 -- came from 0.5*(jf+jr) with no transpose and are
# retained here only so the difference is on the record.
cl3={3:(9.7021,9.6748,9.7023),4:(9.6515,9.6429,9.6898),5:(9.5686,9.6100,9.6648),
     6:(9.4801,9.5855,9.6405),10:(9.2092,9.5484,9.5825),15:(9.0650,9.5430,9.5617),
     20:(9.0206,9.5426,9.5572)}
print(f"  {'n':>4} {'S_fwd':>8} {'claim':>8} {'S_rev':>8} {'claim':>8} {'S_avg':>8} {'claim':>8}")
for n in [3,4,5,6,10,15,20]:
    N=n-1
    Jf=np.asarray(H.pairwise_joint(m,0,N-1,"forward")); Jr=np.asarray(H.pairwise_joint(m,0,N-1,"reverse"))
    Ja=np.asarray(H.orientation_averaged_joint(m,0,N-1))   # reverse term transposed
    print(f"  {n:>4} {ent2(Jf):>8.4f} {cl3[n][0]:>8.4f} {ent2(Jr):>8.4f} {cl3[n][1]:>8.4f} {ent2(Ja):>8.4f} {cl3[n][2]:>8.4f}")

# (sections 9 and 10 removed: they compared Phi_rest against Phi_sum claims using a
#  delta-seeded accumulator.  Superseded by section 11 below, which uses eq-26a seeding.
#  See the review notes accompanying the manuscript.)

print("\n=== 11. Phi_rest: transition count (structural test) ===")
# Phi_sum was removed from the model: only Phi_rest, in which the anchor angle
# is held on its own axis, appears in the manuscript.  The structural test that
# pins the transition count is now the n=3 case, where Phi_rest contains a
# single angle reached by a single transition and must therefore be identical
# to the one-step marginal P(phi_2).
J3=np.asarray(H.anchor_phirest_joint(m,3,"first","forward",True))
r=J3.sum(0)*dφ; r/=r.sum()*dφ
ref2=np.asarray(H.marginal(m,1,"forward"))
d3=float(np.abs(r-ref2).max())
print(f"   n=3 Phi_rest == P(phi_2): max|diff| = {d3:.2e}   "
      f"{'PASS' if d3<1e-12 else '**FAIL**'}")

def _e(p):
    p=np.maximum(np.asarray(p),1e-300); return -float((p*np.log(p)).sum()*dφ)
print(f"   {'n':>4} {'S(Phi_rest)':>12} {'P_cis':>8} {'P_trans':>8}")
mt=np.asarray(H.MASK_TRANS); mc=np.asarray(H.MASK_CIS)
for n in [3,4,5,10,15,20]:
    J=np.asarray(H.anchor_phirest_joint(m,n,"first","forward",True))
    q=J.sum(0)*dφ; q/=q.sum()*dφ
    print(f"   {n:>4} {_e(q):>12.4f} {q[mc].sum()*dφ:>8.4f} {q[mt].sum()*dφ:>8.4f}")
