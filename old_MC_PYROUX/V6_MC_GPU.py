# MC_PYROUX_V8 : montecarlo core + one material + pencil beam + gaussian beam + detector + multimaterial 
# + custom phase function + polar + boucle sur Lz + boucle sur polarisation + boucle sur longueur d'onde + boucle sur densité brouillard
# + version optimisée + parallélisée sur GPU


import math
import numpy as np
from scipy.interpolate import interp1d
import os
import itertools
from numba import cuda
import numba
from numba import uint64, float64, int32

# ==========================================================
# Parametres


polar_inputs = {
    "1000": (1,0,0,0),
    "1001": (1,0,0,1),
    "1010": (1,0,1,0),
    "1100": (1,1,0,0)
}
density_values = [40] #[20,35,47,65,124] #densité pavin small_10
Lambda_values = [9.8]# [0.55, 1.55, 4, 9.8]
LZ_values = [5000]#[100,500,1000,2000,3000,4000,5000] # pavin small_10 et small #variation épaisseur brouillard


Nphoton = 100_000
xSize = 100.0
ySize = 100.0
Nx = Ny = Nz = 100
dx = 2 * xSize / Nx
dy = 2 * ySize / Ny

weightmin = 0.001
chance    = 0.1
eps       = 1e-10
W0        = 2.54
theta_src = 2e-3

xdet = ydet = 0.0
diametre = 15.0
NA       = 0.1

CDFSIZE        = 200
POLAR_TABLE_SIZE = 500
BLOCK_SIZE = 256
N_BLOCKS   = max(1, (Nphoton + BLOCK_SIZE - 1) // BLOCK_SIZE)

# ==========================================================
# I/O CPU


_coeff_cache = {}
_phase_cache = {}
_polar_cache = {}


def load_optical_coeffs(filename):
    if filename in _coeff_cache:
        return _coeff_cache[filename]
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                r = (float(parts[0])*10, float(parts[1])*10,
                     float(parts[2])*10, float(parts[3]))
                _coeff_cache[filename] = r
                return r
    raise ValueError(filename)


def load_phase_cdf(filename):
    if filename in _phase_cache:
        return _phase_cache[filename]
    mu_file, phase_file = [], []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mu_file.append(float(parts[0]))
            phase_file.append(float(parts[1]))
            if len(mu_file) >= 2000:
                break
    theta_deg = np.flip([math.acos(m)*180/math.pi for m in mu_file])
    phase_arr = np.flip(phase_file)
    pfunc = interp1d(theta_deg, phase_arr, kind="linear")
    thetas = [(math.pi/CDFSIZE)*(ik-0.5) for ik in range(1, CDFSIZE+1)]
    pdf = [math.sin(t)*float(pfunc(t*180/math.pi)) for t in thetas]
    S = sum(pdf); pdf = [p/S for p in pdf]
    CDF = np.zeros(CDFSIZE+1, dtype=np.float64)
    cumul = 0.0
    for i, p in enumerate(pdf):
        cumul += p; CDF[i+1] = cumul
    CDF[-1] = 1.0
    _phase_cache[filename] = CDF
    return CDF


def load_polar_tables(filename):
    if filename in _polar_cache:
        return _polar_cache[filename]
    mu_vals, s11_v, s12_v, s33_v, s43_v = [], [], [], [], []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 5: continue
            mu_vals.append(float(parts[0]))
            s11_v.append(float(parts[1])); s12_v.append(float(parts[2]))
            s33_v.append(float(parts[3])); s43_v.append(float(parts[4]))
            if len(mu_vals) >= 2000: break
    theta_vals = np.array([math.acos(m) for m in mu_vals])
    if len(theta_vals) > 1 and theta_vals[1] < theta_vals[0]:
        theta_vals=theta_vals[::-1]; s11_v=s11_v[::-1]; s12_v=s12_v[::-1]
        s33_v=s33_v[::-1]; s43_v=s43_v[::-1]
    grid = np.linspace(0.0, math.pi, POLAR_TABLE_SIZE)
    def interp(vals, f0, f1):
        return interp1d(theta_vals, vals, kind="linear",
                        bounds_error=False, fill_value=(f0,f1))(grid).astype(np.float64)
    r = (interp(s11_v,s11_v[0],s11_v[-1]), interp(s12_v,s12_v[0],s12_v[-1]),
         interp(s33_v,s33_v[0],s33_v[-1]), interp(s43_v,s43_v[0],s43_v[-1]))
    _polar_cache[filename] = r
    return r


# ==========================================================
# Kernel CUDA


@cuda.jit(device=True, inline=True)
def xoro_next(s):
    s0 = s[0]; s1 = s[1]
    result = (s0 + s1) & numba.uint64(0xFFFFFFFFFFFFFFFF)
    s1 ^= s0
    s[0] = (((s0 << numba.uint64(55)) | (s0 >> numba.uint64(9))) ^ s1 ^ (s1 << numba.uint64(14))) & numba.uint64(0xFFFFFFFFFFFFFFFF)
    s[1] = ((s1 << numba.uint64(36)) | (s1 >> numba.uint64(28))) & numba.uint64(0xFFFFFFFFFFFFFFFF)
    return result

@cuda.jit(device=True, inline=True)
def rand_u(s):
    r = xoro_next(s)
    v = float(r >> numba.uint64(11)) * (1.0 / 9007199254740992.0)
    if v <= 0.0: v = 1e-15
    if v >= 1.0: v = 1.0 - 1e-15
    return v

@cuda.jit(device=True, inline=True)
def polar_lut(theta, tab, n):
    t = theta / math.pi * (n - 1)
    i = int(t)
    if i < 0: i = 0
    if i >= n-1: return tab[n-1]
    return tab[i]*(1.0-(t-i)) + tab[i+1]*(t-i)

@cuda.jit(device=True)
def gauss_src(s, w0, div):
    phi_r = 2.0*math.pi*rand_u(s)
    r_mag = w0*math.sqrt(-0.5*math.log(rand_u(s)))
    xt = r_mag*math.cos(phi_r)
    yt = r_mag*math.sin(phi_r)
    phi_d = 2.0*math.pi*rand_u(s)
    phi_div = math.atan(math.tan(div)*math.sqrt(-0.5*math.log(rand_u(s))))
    sp = math.sin(phi_div); cp = math.cos(phi_div)
    ux = sp*math.cos(phi_d); uy = sp*math.sin(phi_d); uz = cp
    if math.fabs(uz) < 1e-10: uz = 1e-10
    x = xt; y = yt; z = 0.0
    return x, y, z, ux, uy, uz

@cuda.jit(device=True)
def cross_t(px, py, pz, ux, uy, uz, ix, iy, iz, dz):
    xmn = -xSize+ix*dx; xmx = xmn+dx
    ymn = -ySize+iy*dy; ymx = ymn+dy
    zmn = iz*dz;         zmx = zmn+dz
    tx = ((xmx-px)/ux if ux>0 else (xmn-px)/ux) if ux!=0 else 1e30
    ty = ((ymx-py)/uy if uy>0 else (ymn-py)/uy) if uy!=0 else 1e30
    tz = ((zmx-pz)/uz if uz>0 else (zmn-pz)/uz) if uz!=0 else 1e30
    return tx, ty, tz

@cuda.jit(device=True)
def scat_pol(s, ux, uy, uz, I, Q, U, V, s11t, s12t, s33t, s43t, nt):
    s11_0 = s11t[0]; I_ref = s11_0*I
    ct = 0.0; phi = 0.0; s11v = 1.0; s12v = 0.0
    for _a in range(100000):
        ct  = 2.0*rand_u(s)-1.0
        th  = math.acos(ct)
        phi = 2.0*math.pi*rand_u(s)
        s11v = polar_lut(th, s11t, nt)
        s12v = polar_lut(th, s12t, nt)
        c2p = math.cos(2.0*phi); s2p = math.sin(2.0*phi)
        It = s11v*I + s12v*(Q*c2p + U*s2p)
        if rand_u(s)*I_ref <= It: break
    st = math.sqrt(max(0.0, 1.0-ct*ct))
    cp = math.cos(phi); sp = math.sin(phi)
    # Rotation Stokes
    c2 = math.cos(2.0*phi); s2 = math.sin(2.0*phi)
    Q2 = Q*c2+U*s2; U2 = -Q*s2+U*c2; Q=Q2; U=U2
    s33v = polar_lut(math.acos(ct), s33t, nt)
    s43v = polar_lut(math.acos(ct), s43t, nt)
    Io=s11v*I+s12v*Q; Qo=s12v*I+s11v*Q; Uo=s33v*U+s43v*V; Vo=-s43v*U+s33v*V
    I=Io; Q=Qo; U=Uo; V=Vo
    uz_old = uz
    if math.fabs(uz)<1.0:
        tmp = math.sqrt(1.0-uz*uz)
        ux2 = st*(ux*uz*cp-uy*sp)/tmp+ux*ct
        uy2 = st*(uy*uz*cp+ux*sp)/tmp+uy*ct
        uz2 = -st*cp*tmp+uz*ct
    else:
        sgn = 1.0 if uz>0 else -1.0
        ux2=st*cp; uy2=st*sp; uz2=ct*sgn
    ux=ux2; uy=uy2; uz=uz2
    den = math.sqrt(max(0.0,1.0-ct*ct))*math.sqrt(max(0.0,1.0-uz*uz))
    if den < 1e-12:
        ci = 0.0
    else:
        ci = ((-uz*ct+uz_old) if phi<=math.pi else (uz*ct-uz_old))/den
        if ci > 1.0: ci=1.0
        if ci <-1.0: ci=-1.0
    si = math.sqrt(max(0.0,1.0-ci*ci))
    c2psi=2.0*ci*ci-1.0; s2psi=2.0*si*ci
    Qf=Q*c2psi-U*s2psi; Uf=Q*s2psi+U*c2psi
    if I>0.0:
        Q=Qf/I; U=Uf/I; V=V/I; I=1.0
    else:
        I=1.0; Q=0.0; U=0.0; V=0.0
    return ux, uy, uz, I, Q, U, V


@cuda.jit
def mc_kernel(zSize, dz, mua, mus,
              s11t, s12t, s33t, s43t,
              I_in, Q_in, U_in, V_in,
              seeds,
              oR, oT, oA,
              oRI, oRQ, oRU, oRV,
              oTI, oTQ, oTU, oTV,
              oDet, oDI, oDQ, oDU, oDV,
              det_x, det_y, det_z, det_d, det_NA,
              n_ph):

    tid = cuda.grid(1)
    if tid >= n_ph: return

    nt = s11t.shape[0]
    rng = cuda.local.array(2, dtype=numba.uint64)
    rng[0] = seeds[tid, 0]
    rng[1] = seeds[tid, 1]

    px, py, pz, ux, uy, uz = gauss_src(rng, W0, theta_src)
    I=I_in; Q=Q_in; U=U_in; V=V_in
    weight = 1.0
    step_left = -math.log(rand_u(rng))

    for _main in range(2_000_000):
        if weight <= 0.0: break

        for _prop in range(1_000_000):
            if step_left <= 0.0: break
            ix = int((px+xSize)/dx)
            iy = int((py+ySize)/dy)
            iz = int(pz/dz)
            tx, ty, tz = cross_t(px, py, pz, ux, uy, uz, ix, iy, iz, dz)
            s_sc = step_left/mus if mus>0.0 else 1e30
            s = min(s_sc, min(tx, min(ty, tz)))

            absorb = -weight*math.expm1(-mua*s)
            weight -= absorb

            if math.fabs(s-tx) < eps:
                px = -xSize+(ix+1)*dx+eps if ux>0 else -xSize+ix*dx-eps
            else:
                px += s*ux
            if math.fabs(s-ty) < eps:
                py = -ySize+(iy+1)*dy+eps if uy>0 else -ySize+iy*dy-eps
            else:
                py += s*uy
            if math.fabs(s-tz) < eps:
                pz = (iz+1)*dz+eps if uz>0 else iz*dz-eps
            else:
                pz += s*uz

            if math.fabs(s-s_sc) < eps:
                step_left = 0.0
            else:
                step_left -= s*mus

            # Sorties boite
            if pz < 0.0:
                cuda.atomic.add(oR,  0, weight)
                cuda.atomic.add(oRI, 0, I*weight); cuda.atomic.add(oRQ, 0, Q*weight)
                cuda.atomic.add(oRU, 0, U*weight); cuda.atomic.add(oRV, 0, V*weight)
                weight = 0.0; break
            elif pz > zSize:
                cuda.atomic.add(oT,  0, weight)
                cuda.atomic.add(oTI, 0, I*weight); cuda.atomic.add(oTQ, 0, Q*weight)
                cuda.atomic.add(oTU, 0, U*weight); cuda.atomic.add(oTV, 0, V*weight)
                # Detecteur
                if math.fabs(uz) > 1e-12:
                    t_d = (det_z - pz + (pz-zSize))/uz
                    xi = px+t_d*ux; yi = py+t_d*uy
                    r2 = math.sqrt((xi-det_x)**2+(yi-det_y)**2)
                    if r2 <= det_d/2.0:
                        th_max = math.asin(det_NA)
                        norm_u = math.sqrt(ux*ux+uy*uy+uz*uz)
                        if math.acos(math.fabs(uz)/norm_u) <= th_max:
                            cuda.atomic.add(oDet, 0, weight)
                            cuda.atomic.add(oDI, 0, I*weight); cuda.atomic.add(oDQ, 0, Q*weight)
                            cuda.atomic.add(oDU, 0, U*weight); cuda.atomic.add(oDV, 0, V*weight)
                weight = 0.0; break
            elif px < -xSize or px > xSize or py < -ySize or py > ySize:
                cuda.atomic.add(oA, 0, weight)
                weight = 0.0; break

        if weight <= 0.0: break

        # Roulette russe
        if weight < weightmin:
            if rand_u(rng) <= chance:
                weight /= chance
            else:
                cuda.atomic.add(oA, 0, weight)
                weight = 0.0; break

        # Diffusion polarisee
        ux, uy, uz, I, Q, U, V = scat_pol(
            rng, ux, uy, uz, I, Q, U, V,
            s11t, s12t, s33t, s43t, nt)
        step_left = -math.log(rand_u(rng))


# ==========================================================
# Host


def run_simulation_gpu(args):
    zSize, lambada, dense, pol_name, stokes = args
    dz = zSize / Nz; zdet = zSize

    coeff_file = f"pavin_oct_alb/coeff_potique_Pav_al_small_10_v_{dense}_l_{lambada}_nm.txt"
    phase_file = f"pavin_oct_alb/ram_function_phases_Pav_al_small_10_v_{dense}_l_{lambada}_nm.txt"

    _mua_f, _mut_f, _mus_f, _g_f = load_optical_coeffs(coeff_file)
    s11_np, s12_np, s33_np, s43_np = load_polar_tables(phase_file)

    mua_val = np.float64(_mua_f )
    mus_val = np.float64(_mus_f )
    I_in, Q_in, U_in, V_in = [np.float64(v) for v in stokes]

    rng_np = np.random.default_rng()
    seeds_np = rng_np.integers(1, 2**63, size=(Nphoton, 2), dtype=np.uint64)

    d_s11 = cuda.to_device(s11_np); d_s12 = cuda.to_device(s12_np)
    d_s33 = cuda.to_device(s33_np); d_s43 = cuda.to_device(s43_np)
    d_seeds = cuda.to_device(seeds_np)

    def z1(): return cuda.to_device(np.zeros(1, dtype=np.float64))
    oR=z1(); oT=z1(); oA=z1()
    oRI=z1(); oRQ=z1(); oRU=z1(); oRV=z1()
    oTI=z1(); oTQ=z1(); oTU=z1(); oTV=z1()
    oDet=z1(); oDI=z1(); oDQ=z1(); oDU=z1(); oDV=z1()

    print(f"  GPU > LZ={zSize} POL={pol_name} lambda={lambada} dense={dense}  [{N_BLOCKS}x{BLOCK_SIZE}]")

    mc_kernel[N_BLOCKS, BLOCK_SIZE](
        np.float64(zSize), np.float64(dz), mua_val, mus_val,
        d_s11, d_s12, d_s33, d_s43,
        I_in, Q_in, U_in, V_in, d_seeds,
        oR, oT, oA, oRI, oRQ, oRU, oRV, oTI, oTQ, oTU, oTV,
        oDet, oDI, oDQ, oDU, oDV,
        np.float64(xdet), np.float64(ydet), np.float64(zdet),
        np.float64(diametre), np.float64(NA), np.int32(Nphoton)
    )
    cuda.synchronize()

    def g(d): return float(d.copy_to_host()[0])
    cR=g(oR); cT=g(oT); cA=g(oA)
    R_I=g(oRI); R_Q=g(oRQ); R_U=g(oRU); R_V=g(oRV)
    T_I=g(oTI); T_Q=g(oTQ); T_U=g(oTU); T_V=g(oTV)
    D=g(oDet); D_I=g(oDI); D_Q=g(oDQ); D_U=g(oDU); D_V=g(oDV)

    nom = f"polar_result_small_10_v40/log_GPU_v{dense}_l{lambada}_z{zSize}_pol{pol_name}.txt"
    _write_results(nom, Nphoton, cR, cT, cA,
                   R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V,
                   D, D_I, D_Q, D_U, D_V)
    return nom


def _write_results(nom, Nphoton, cR, cT, cA,
                   R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V,
                   D, D_I, D_Q, D_U, D_V):
    R_rate = cR/Nphoton; T_rate = cT/Nphoton
    A_rate = 1.0-R_rate-T_rate; C_rate = D/Nphoton
    with open(nom, "w", encoding="utf-8") as f:
        f.write("="*66+"\n  RESULTATS MONTE CARLO GPU\n"+"="*66+"\n")
        f.write(f"\nPhotons emis : {Nphoton}\n")
        f.write(f"Reflexion   : {cR:.4f}  ({R_rate:.6f})\n")
        f.write(f"Transmission: {cT:.4f}  ({T_rate:.6f})\n")
        f.write(f"Absorption  : {cA:.4f}  ({A_rate:.6f})\n")
        f.write(f"Detecteur   : {D:.4f}  ({C_rate:.6f}  = {100*C_rate:.4f}%)\n")
        for lbl,sI,sQ,sU,sV in [("REFLEXION",R_I,R_Q,R_U,R_V),
                                  ("TRANSMISSION",T_I,T_Q,T_U,T_V),
                                  ("DETECTEUR",D_I,D_Q,D_U,D_V)]:
            f.write(f"\n--- {lbl} ---\nI={sI:.6f} Q={sQ:.6f} U={sU:.6f} V={sV:.6f}\n")
            if sI>0:
                DOP  = math.sqrt(sQ**2+sU**2+sV**2)/sI
                LDOP = math.sqrt(sQ**2+sU**2)/sI
                CDOP = abs(sV)/sI
                f.write(f"DOP={DOP:.6f}  LDOP={LDOP:.6f}  CDOP={CDOP:.6f}\n")
        f.write("="*66+"\nFin\n"+"="*66+"\n")
    print("  Ecrit :", nom)


# ==========================================================
# MAIN


if __name__ == "__main__":
    try:
        cuda.detect()
    except Exception as e:
        print(f"[ERREUR] GPU CUDA non detecte : {e}")
        raise SystemExit(1)

    all_tasks = [(z, lam, d, pname, stokes)
                 for z, lam, d, (pname, stokes)
                 in itertools.product(LZ_values, Lambda_values,
                                      density_values, polar_inputs.items())]
    total = len(all_tasks)
    print(f"\n{'='*60}\n  {total} simulations x {Nphoton} photons -- GPU CUDA")
    print(f"  {N_BLOCKS} blocs x {BLOCK_SIZE} threads/bloc\n{'='*60}\n")

    results = [run_simulation_gpu(t) for t in all_tasks]
    print(f"\nTermine : {len(results)} fichiers.")

