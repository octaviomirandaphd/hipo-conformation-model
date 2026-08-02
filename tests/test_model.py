"""Regression tests for the corrected JAX core."""
import numpy as np, jax.numpy as jnp
from hipo import core as H
m, dφ = H.MODEL, H.DPHI
Pp, P_T, Tf = np.asarray(m.P_phi_T), np.asarray(m.P_T), np.asarray(m.T_fwd)
ok = lambda c: "PASS" if c else "**FAIL**"

def exact_pair(k):
    W = P_T[:,None]*np.linalg.matrix_power(Tf,k); J = Pp.T@(W@Pp)
    return J/(J.sum()*dφ**2)

print("=== 1. normalisation ===")
for n,i,j in [(3,0,1),(5,0,3),(20,0,18)]:
    J=np.asarray(H.pairwise_joint(m,i,j,"forward"))
    print(f"  pairwise n={n:<3} integral = {J.sum()*dφ**2:.15f}  {ok(abs(J.sum()*dφ**2-1)<1e-12)}")

print("\n=== 2. correlation preserved (must be non-zero) ===")
def MI(J):
    J=np.asarray(J,dtype=np.float64); J=J/(J.sum()*dφ**2)
    px,py=J.sum(1)*dφ,J.sum(0)*dφ; ind=np.outer(px,py); msk=J>1e-300
    return float((J[msk]*np.log(J[msk]/ind[msk])).sum()*dφ**2)
for n in [3,5,20]:
    N=n-1; mi=MI(H.pairwise_joint(m,0,N-1,"forward"))
    print(f"  n={n:<3} I(φ1;φ{N}) = {mi:.6e}  {ok(mi>1e-6)}")
print("  unnormalised variant keeps the same correlation:")
for n in [3,20]:
    N=n-1
    a=MI(H.pairwise_joint(m,0,N-1,"forward",normalise=True))
    b=MI(H.pairwise_joint(m,0,N-1,"forward",normalise=False))
    print(f"    n={n:<3} norm {a:.4e} vs unnorm {b:.4e}  {ok(abs(a-b)<1e-9)}")

print("\n=== 3. anchor/Φ_rest off-by-one fixed ===")
A=np.asarray(H.anchor_phirest_joint(m,3,"first","forward"))
print(f"  n=3 first vs exact P(φ1,φ2)  [1 transition ] max|Δ| = {np.abs(A-exact_pair(1)).max():.3e}  {ok(np.abs(A-exact_pair(1)).max()<1e-12)}")
print(f"  n=3 first vs old over-propagated [2 trans]   max|Δ| = {np.abs(A-exact_pair(2)).max():.3e}  {ok(np.abs(A-exact_pair(2)).max()>1e-6)}")
B=np.asarray(H.anchor_phirest_joint(m,3,"last","forward"))
print(f"  n=3 last  vs exact P(φ2,φ1)                  max|Δ| = {np.abs(B-exact_pair(1).T).max():.3e}  {ok(np.abs(B-exact_pair(1).T).max()<1e-12)}")

print("\n=== 4. anchor marginal matches direct propagation ===")
for n in [3,5,10,20]:
    N=n-1
    Jf=np.asarray(H.anchor_phirest_joint(m,n,"first","forward"))
    d1=np.abs(Jf.sum(1)*dφ - np.asarray(H.marginal(m,0,"forward"))).max()
    Jl=np.asarray(H.anchor_phirest_joint(m,n,"last","forward"))
    d2=np.abs(Jl.sum(1)*dφ - np.asarray(H.marginal(m,N-1,"forward"))).max()
    print(f"  n={n:<3} first {d1:.2e}  last {d2:.2e}  {ok(max(d1,d2)<1e-12)}")

print("\n=== 5. Bayes relation at one step ===")
P_next=np.asarray(m.T_fwd.T@m.P_T)
jf=P_T[:,None]*np.asarray(m.T_fwd); jr=P_next[:,None]*np.asarray(m.T_rev)
d=np.abs(jf-jr.T).max()
print(f"  max|P(Ti)P(Tj|Ti) - P(Tj)P(Ti|Tj)| = {d:.3e}  {ok(d<1e-12)}")

print("\n=== 6. P(T) is NOT stationary (documented limitation) ===")
d=np.abs(P_T@np.asarray(m.T_fwd)-P_T).max()
print(f"  max|P_T @ T_fwd - P_T| = {d:.6f}   -> T_rev**s is exact only at s=1")
print("   step   max|Bayes-exact reverse - T_rev**s|")
Tr=np.asarray(m.T_rev)
for s in range(1,6):
    Ff=np.linalg.matrix_power(Tf,s); Pf=P_T@Ff
    true=(Ff*P_T[:,None]/Pf[None,:]).T
    print(f"     {s}     {np.abs(true-np.linalg.matrix_power(Tr,s)).max():.3e}")

print("\n=== 7. orientation-average uses the mirrored index (eq 6) ===")
for n in [5,20]:
    N=n-1
    a=np.asarray(H.avg_marginal(m,n,0)); b=np.asarray(H.avg_marginal(m,n,N-1))
    fa=np.asarray(H.marginal(m,0,"forward")); ra=np.asarray(H.marginal(m,N-1,"reverse"))
    ref=0.5*(fa+ra); ref/=ref.sum()*dφ
    print(f"  n={n:<3} k=0 matches ½[fwd(0)+rev(N-1)] : {np.abs(a-ref).max():.2e}  {ok(np.abs(a-ref).max()<1e-12)}")
    print(f"  n={n:<3} avg(k=0) vs avg(k=N-1) differ  : {np.abs(a-b).max():.2e}  {ok(np.abs(a-b).max()>1e-6)}")

print("\n=== 8. funneling to opposite chain ends ===")
for nm,M_ in [("T_fwd",Tf),("T_rev",Tr)]:
    w,v=np.linalg.eig(M_.T); k=np.argmin(abs(w-1)); pi=np.real(v[:,k]); pi/=pi.sum()
    print(f"  {nm}: stationary -> {H.TAUT_NAMES[int(np.argmax(pi))]}  {np.array2string(pi,precision=6)}")
