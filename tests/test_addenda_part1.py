import numpy as np
from hipo import core as H
np.set_printoptions(precision=6, suppress=True, linewidth=150)
m=H.MODEL; Pp=np.asarray(m.P_phi_T); P_T=np.asarray(m.P_T)
Tf=np.asarray(m.T_fwd); Tr=np.asarray(m.T_rev); dφ=H.DPHI; NM=H.TAUT_NAMES

print("="*72); print("1. TRANSITION MATRICES"); print("="*72)
print("T_fwd ="); print(Tf)
print("T_rev ="); print(Tr)
print("\nclaimed T_rev rows (cis_trans_interpretation):")
print("  Ta [0.99999980, 1.96e-7, 0, 0] ; Tb/Td [0, 0, 0.40507, 0.59493]")
print("claimed (methods_paragraph): T_rev[Td->Tc]=0.4051  T_rev[Td->Td]=0.5949")
print(f"computed  T_rev[Td->Tc]={Tr[3,2]:.5f}  T_rev[Td->Td]={Tr[3,3]:.5f}")
print(f"computed  T_rev[Ta->Ta]={Tr[0,0]:.8f}  T_rev[Ta->Tc]={Tr[0,2]:.5f}")
print("claimed (kstar addendum): p_R = T_R[Ta->Tc] = 0.265")

print("\n"+"="*72); print("2. EIGENVALUES / STATIONARY"); print("="*72)
for nm,M_ in [("T_fwd",Tf),("T_rev",Tr)]:
    ev=np.linalg.eigvals(M_); ev=np.sort_complex(ev)[::-1]
    w,v=np.linalg.eig(M_.T); k=np.argmin(abs(w-1)); pi=np.real(v[:,k]); pi/=pi.sum()
    print(f"{nm}: eigenvalues {np.round(np.real(ev),6)}   stationary {np.round(pi,6)} -> {NM[int(np.argmax(pi))]}")
print("claim (Claude_polymer §4): lambda2 = 0.7353, lambda3 = lambda4 = 0")
print("claim (methods_addendum) : lambda2(T_fwd)=0.7353 > lambda2(T_rev)=0.5949")
print("claim (cis_trans §2)     : 'Both networks converge to the SAME equilibrium {0.529,0.191,0.280}'")
print("claim (Claude_polymer §5): 'stationary distribution of forward matrix converges to P(Td)~1.0'")

print("\n"+"="*72); print("3. ESCAPE PROBABILITIES"); print("="*72)
print(f"  T_fwd[Ta->Tc] = {Tf[0,2]:.5f}      T_fwd[Td->Tc] = {Tf[3,2]:.5e}")
print(f"  T_rev[Ta->Tc] = {Tr[0,2]:.5e}   T_rev[Td->Tc] = {Tr[3,2]:.5f}")
print("  manuscript/kstar : p_F = 0.405 (Td->Tc), p_R = 0.265 (Ta->Tc)")
print("  methods_paragraph: p_fwd = 0.26, p_rev = 0.40")
print("  cis_trans        : p_fwd = 0.26, p_rev = 0.40")

print("\n"+"="*72); print("4. k*  —  BOTH REPORTED CONVENTIONS"); print("="*72)
pA,pB = 0.26469596, 0.40507   # Ta->Tc  and  Td->Tc
f_ct = np.log(1-pB)/(np.log(1-pA)+np.log(1-pB))
print(f"  cis_trans eq33  k*/N = ln(1-0.40)/[ln(1-0.74... )] = {f_ct:.4f}   (claimed 0.629)")
f_ks = np.log(1-pA)/(np.log(1-pA)+np.log(1-pB))
print(f"  kstar  (k*-1)/(N-1) = {f_ks:.4f}   (claimed 0.372)     sum = {f_ct+f_ks:.4f}")
print(f"\n  {'n':>4} {'N':>4} {'cis_trans k*=f*N':>18} {'kstar k*':>12} {'manuscript eq':>15}")
for n in [3,4,5,10,15,20,30,50]:
    N=n-1
    k_ct = f_ct*N
    k_ks = f_ks*(N-1)+1
    k_ms = (np.log(1-0.405)+(n-1)*np.log(1-0.265))/(np.log(1-0.405)+np.log(1-0.265))
    print(f"  {n:>4} {N:>4} {k_ct:>18.3f} {k_ks:>12.3f} {k_ms:>15.3f}")
print("  cis_trans table claims n=20 -> 11.95 ; kstar table claims n=20 -> 7.694")

print("\n"+"="*72); print("5. BELIEF CROSSINGS (mixture vs product)"); print("="*72)
def alpha(k,N): return P_T@np.linalg.matrix_power(Tf,k-1)
def beta(k,N):  return P_T@np.linalg.matrix_power(Tr,N-k)
def cross(n,rule):
    N=n-1; ks=np.arange(1,N+1); d=[]
    for k in ks:
        a,b=alpha(k,N),beta(k,N)
        g = 0.5*(a+b) if rule=="mix" else (a*b/P_T)
        g=g/g.sum(); d.append(g[0]-g[3])
    d=np.array(d)
    s=np.where(np.sign(d[:-1])!=np.sign(d[1:]))[0]
    if len(s)==0: return np.nan
    i=s[0]; return ks[i]+d[i]/(d[i]-d[i+1])
print(f"  {'n':>4} {'msg crossing':>14} {'mixture':>10} {'(k-1)/(N-1)':>13} {'product':>10}   claimed(mix/prod)")
claim={6:(2.488,1.885,0.221,2.354),10:(3.975,3.381,0.298,3.906),15:(5.835,5.234,0.326,5.805),
       20:(7.694,7.083,0.338,7.680),30:(11.413,10.815,0.351,11.414),50:(18.851,18.227,0.359,18.834)}
for n in [6,10,15,20,30,50]:
    N=n-1; km=f_ks*(N-1)+1; cm=cross(n,"mix"); cp=cross(n,"prod")
    print(f"  {n:>4} {km:>14.3f} {cm:>10.3f} {(cm-1)/(N-1):>13.3f} {cp:>10.3f}   {claim[n]}")

print("\n"+"="*72); print("6. Tc POPULATIONS (kstar addendum table)"); print("="*72)
print(f"  {'n':>4} {'terminus':>10} {'interior min':>13} {'mean mixture':>13} {'mean product':>13}   claimed")
claimTc={6:(0.123,0.085,0.100,0.046),10:(0.103,0.040,0.063,0.015),
         15:(0.097,0.015,0.042,0.004),20:(0.096,0.006,0.031,0.001)}
for n in [6,10,15,20]:
    N=n-1; gm=[];gp=[]
    for k in range(1,N+1):
        a,b=alpha(k,N),beta(k,N)
        x=0.5*(a+b); x/=x.sum(); gm.append(x[2])
        y=a*b/P_T; y/=y.sum(); gp.append(y[2])
    gm=np.array(gm); gp=np.array(gp)
    print(f"  {n:>4} {gm[0]:>10.4f} {gm[1:-1].min():>13.4f} {gm.mean():>13.4f} {gp.mean():>13.4f}   {claimTc[n]}")
print(f"  claim: gamma_1(Tc) ~ 0.5*P(Tc) = {0.5*P_T[2]:.4f}  (doc says 0.0953, computed 0.0957)")
