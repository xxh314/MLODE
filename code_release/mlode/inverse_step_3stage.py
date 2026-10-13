"""Three-stage inverse: larvae, pupae, adults.

    L' = b(t) A (1 - k(t)(L+P)) - dL(t) L
    P' = dL(t) L - dP(t) P
    A' = (1/2) pEA(t) dP(t) P - mu(t) A          with the observed count x = A - A0

Inverting for k from the adult series, as in the manuscript:

    A  = x + A0
    P  = 2 ( x' + mu A ) / ( pEA dP )
    L  = ( P' + dP P ) / dL
    k  = [ 1 - ( L' + dL L ) / ( b A ) ] / ( L + P )

This needs the THIRD derivative of the smoothed count series, where the two-stage model stops at the
second. The pchip spline used in the two-stage pipeline is only C1, so its third derivative is piecewise
constant and flips sign on about a fifth of days at any smoothing width. A quintic smoothing spline is used
here instead: it is C4, so the third derivative is continuous and piecewise quadratic.

A0 is chosen to keep BOTH immature stages positive, which is two constraints rather than the one the
two-stage model has. The P > 0 constraint has the same closed form as before; the L > 0 constraint is
solved numerically on top of it.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
import inverse_step_2stage as inv

def _deriv(days, x, s_rel=1.0, k_spline=5):
    """quintic smoothing spline and its first three derivatives. s_rel scales the smoothing factor:
    s = s_rel * n * var(x) * 1e-3; larger means smoother and flatter high derivatives."""
    n=len(x); s=max(s_rel*n*np.var(x)*1e-3, 1e-12)
    sp=UnivariateSpline(days, x, k=k_spline, s=s)
    return sp, sp.derivative(1), sp.derivative(2), sp.derivative(3)

def stages(days, x, A0, R):
    """L, P, A and the intermediate derivatives at A0, given a fitted spline set"""
    sp,d1,d2,d3=R["spl"]
    t=np.minimum(days, R["tmax"])
    mu=np.asarray(R["mu"](t),float); pEA=np.asarray(R["pEA"](t),float)
    dP=np.asarray(R["dP"](t),float); dL=np.asarray(R["dL"](t),float); b=np.asarray(R["b"](t),float)
    xs=sp(days); x1=d1(days); x2=d2(days); x3=d3(days)
    A=xs+A0
    num=x1+mu*A
    P=2.0*num/(pEA*dP)
    # P' by differentiating the expression, using derivatives of the thermal splines
    mu1=np.asarray(R["mu"].derivative(1)(t),float); pEA1=np.asarray(R["pEA"].derivative(1)(t),float)
    dP1=np.asarray(R["dP"].derivative(1)(t),float); dL1=np.asarray(R["dL"].derivative(1)(t),float)
    num1=x2+mu1*A+mu*x1
    den=pEA*dP; den1=pEA1*dP+pEA*dP1
    P1=2.0*(num1*den-num*den1)/den**2
    L=(P1+dP*P)/dL
    num2=x3+mu1*x1+mu1*x1+ (0.0)  # d/dt of (x2 + mu1 A + mu x1); mu2 omitted, thermal curvature is negligible here
    num2=x3+2*mu1*x1+mu*x2
    den2=2*pEA1*dP1
    P2=2.0*((num2*den-num*den2)*den**2-(num1*den-num*den1)*2*den*den1)/den**4
    L1=((P2+dP1*P+dP*P1)*dL-(P1+dP*P)*dL1)/dL**2
    return dict(A=A,P=P,L=L,P1=P1,L1=L1,b=b,dL=dL,dP=dP,mu=mu,pEA=pEA)

def a0_rule3(days, x, R, mult=2.0, cap_scan=2000.0):
    """smallest A0 keeping P > 0 and L > 0 everywhere, then multiplied. Returns (A0, m_P, m_L)."""
    sp,d1,d2,d3=R["spl"]; t=np.minimum(days,R["tmax"])
    mu=np.asarray(R["mu"](t),float); xs=sp(days); x1=d1(days)
    mP=float(np.nanmax(-(x1/np.maximum(mu,1e-9)+xs)))        # closed form: P > 0
    lo=max(mP,0.0)
    def okL(a):
        st=stages(days,x,a,R); return np.all(np.isfinite(st["L"])) and np.nanmin(st["L"])>0
    if okL(lo*1.001+1e-6): mL=lo
    else:
        hi=max(lo,1.0)
        while hi<cap_scan and not okL(hi): hi*=1.6
        if hi>=cap_scan: return np.nan,mP,np.nan
        a,b_=lo,hi
        for _ in range(40):
            m=(a+b_)/2
            if okL(m): b_=m
            else: a=m
        mL=b_
    m=max(mP,mL)
    return float(mult*np.ceil(m)) if m>0 else 0.0, float(mP), float(mL)

def recover3(trap, species, zeros="keep", cutoff=None, smooth="centred", window=28, mult=2.0,
             start="2017-01-01", floor=0.5, s_rel=1.0):
    """full three-stage chain for one trap -> dict(days,x,k,L,P,A0,...) or None"""
    st=inv.trap_station().get(trap)
    if st is None: return None
    t,y=inv.trap_observations(trap,species,zeros=zeros,floor=floor,cutoff=cutoff)
    days,x=inv.smooth_series(t,y,smooth=smooth,window=window,start=start)
    if days is None or len(days)<60: return None
    Rt=inv.rates(st,species); days=days[days<=Rt.tmax]; x=x[:len(days)]
    if not hasattr(Rt,"dL"): return None
    R=dict(spl=_deriv(days.astype(float),x,s_rel=s_rel),tmax=Rt.tmax,
           mu=Rt.mu,pEA=Rt.pEA,dP=Rt.dP,dL=Rt.dL,b=Rt.b)
    A0,mP,mL=a0_rule3(days.astype(float),x,R,mult=mult)
    if not np.isfinite(A0): return None
    s=stages(days.astype(float),x,A0,R)
    k=(1.0-(s["L1"]+s["dL"]*s["L"])/(s["b"]*s["A"]))/(s["L"]+s["P"])
    return dict(trap=trap,station=st,days=days,x=x,k=k,L=s["L"],P=s["P"],A=s["A"],A0=A0,
                m_P=mP,m_L=mL,mean_count=float(y[y>0].mean()) if (y>0).any() else 0.0,n_obs=len(t))

def forward3(R3, k_of_t, A0, days, y0=None):
    """three-stage forward simulation. R3 is an inverse.Rates carrying dL and dP.
    L' = b A (1 - k(L+P)) - dL L ; P' = dL L - dP P ; A' = (1/2) pEA dP P - mu A ; count = A - A0."""
    from scipy.integrate import odeint
    tm=R3.tmax
    def rhs(v,tt):
        L,Pp,A=v; t=min(tt,tm); kk=float(k_of_t(tt))
        return [float(R3.b(t))*A*(1-kk*(L+Pp))-float(R3.dL(t))*L,
                float(R3.dL(t))*L-float(R3.dP(t))*Pp,
                0.5*float(R3.pEA(t))*float(R3.dP(t))*Pp-float(R3.mu(t))*A]
    v0=y0 if y0 is not None else [A0,A0,A0]
    sol=odeint(rhs,v0,days)
    return sol[:,0],sol[:,1],sol[:,2],sol[:,2]-A0
