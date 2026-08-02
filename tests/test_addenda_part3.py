import numpy as np
from hipo import core as H
m=H.MODEL; Pp=np.asarray(m.P_phi_T); P_T=np.asarray(m.P_T)
Tf=np.asarray(m.T_fwd); Tr=np.asarray(m.T_rev); dφ=H.DPHI

print("=== A. methods_paragraph T_rev formula, taken literally ===")
print("   doc:  T_rev[Ti,Tj] = T_fwd[Tj,Ti] * P(Ti)/P(Tj)      <- equilibrium P in denominator")
print("   code: T_rev[j,i]   = T_fwd[i,j]  * P_T[i]/P_next[j]  <- P_next = P_T @ T_fwd")
doc =(Tf*P_T[:,None]/P_T[None,:]).T
print("   doc-literal T_rev row sums :", np.round(doc.sum(1),6), " <- must be 1 to be a kernel")
print("   code        T_rev row sums :", np.round(Tr.sum(1),6))
print(f"   max|doc - code| = {np.abs(doc-Tr).max():.4f}")

print("\n=== B. Claude_polymer §6: unnormalised peak growth k=0 -> k=9 ===")
y0=P_T@Pp; y9=(P_T@np.linalg.matrix_power(Tf,9))@Pp
print(f"   peak(k=9)/peak(k=0) = {y9.max()/y0.max():.3f}   (claim ~2.3x)")

print("\n=== C. Claude_polymer §7-8: 'joint entropy INCREASES with chain length' ===")
def ent2(J):
    J=np.asarray(J); J=J/(J.sum()*dφ**2); p=np.maximum(J,1e-300)
    return -float((p*np.log(p)).sum()*dφ**2)
for n in [3,10,20]:
    N=n-1
    print(f"   n={n:<3} S_fwd = {ent2(H.pairwise_joint(m,0,N-1,'forward')):.4f}")
print("   -> corrected S_fwd DECREASES monotonically; entropy_correction_addendum agrees,")
print("      Claude_polymer_physical_interpretation §7/§8 still says it increases.")

print("\n=== D. Tc bridge extent, 20-mer, delta = 0.10 (methods_addendum) ===")
TF,TR=Tr,Tf   # addendum convention
n=20; N=n-1
ks=np.arange(1,N+1)
d=[]
for k in ks:
    A=P_T@np.linalg.matrix_power(TF,k-1); B=P_T@np.linalg.matrix_power(TR,N-k)
    g=0.5*(A+B); g/=g.sum(); d.append(abs(g[0]-g[3]))
d=np.array(d); inside=ks[d<0.10]
print(f"   positions satisfying |P(Ta)-P(Td)| < 0.10 : k = {inside.min()} .. {inside.max()}  ({len(inside)} dimers)")
print(f"   claim: 'k = 4 to k = 14 (10 dimers wide)'  -> k=4..14 inclusive is 11 positions")

print("\n=== E. chain-end saturation (methods_addendum: Ta by n~12, Td by n~18) ===")
prev_a=prev_d=None
for n in range(4,31):
    N=n-1
    gN=P_T@np.linalg.matrix_power(TF,N-1); g1=P_T@np.linalg.matrix_power(TR,N-1)
    a=0.5*(gN[0]+P_T[0]); dd=0.5*(g1[3]+P_T[3])
    if prev_a is not None:
        if abs(a-prev_a)<1e-3 and abs(a-prev_a)>0: pass
    prev_a,prev_d=a,dd
    if n in (10,12,15,18,20,25,30):
        print(f"   n={n:<3} Ta at Ta-end = {a:.5f}   Td at Td-end = {dd:.5f}")
