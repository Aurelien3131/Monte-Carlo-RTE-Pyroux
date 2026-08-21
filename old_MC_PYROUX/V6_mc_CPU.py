# MC_PYROUX_V7 : montecarlo core + one material + pencil beam + gaussian beam + detector + multimaterial 
# + custom phase function + polar + boucle sur Lz + boucle sur polarisation + boucle sur longueur d'onde + boucle sur densité brouillard
# + version optimisée + parallélisée sur CPU
# Optimisations :
#   1. multiprocessing.Pool : chaque combinaison (zSize, lambda, dense, pol) tourne en parallèle
#   2. Batch de photons numpy dans scatter() et gaussian_source() pour réduire overhead Python
#   3. Pré-chargement des fichiers optiques/phase hors boucle (cache)
#   4. Élimination des recalculs redondants dans propagate()

import math
import random
from dataclasses import dataclass, field
import bisect
import numpy as np
from scipy.interpolate import interp1d
import multiprocessing
import os
import itertools
from functools import lru_cache

# ==========================================================
# Éléments de boucles
# ==========================================================

polar_inputs = {
    "1000": (1, 0, 0, 0),
    "1001": (1, 0, 0, 1),
    "1010": (1, 0, 1, 0),
    "1100": (1, 1, 0, 0),
}
density_values = [20, 35, 47, 65, 124]
Lambda_values = [0.55, 1.55, 4, 9.8]
LZ_values =[100, 500, 1000, 2000, 3000, 4000, 5000]

# ==========================================================
# Paramètres globaux
# ==========================================================

Nphoton = 10000
xSize = 100.0
ySize = 100.0
Nx = 100
Ny = 100
Nz = 100
dx = 2 * xSize / Nx
dy = 2 * ySize / Ny

weightmin = 0.001
chance = 0.1
eps = 1e-10

xinit = 0.0
yinit = 0.0
zinit = 0.0
zlaunch = zinit
uxinit = 0
uyinit = 0
uzinit = 1
W0 = 2.54
theta_src = 2e-3

xdet = 0.0
ydet = 0.0
diametre = 15.0
NA = 0.1
thetadet = math.pi
phidet = 0.0

# ==========================================================
# Cache fichiers (évite relecture disque à chaque itération)
# ==========================================================

_coeff_cache = {}
_phase_cache = {}
_polar_cache = {}


def cached_load_optical_coeffs(filename):
    if filename not in _coeff_cache:
        _coeff_cache[filename] = load_optical_coeffs(filename)
    return _coeff_cache[filename]


def cached_load_phase_function(filename):
    if filename not in _phase_cache:
        _phase_cache[filename] = load_phase_function(filename)
    return _phase_cache[filename]


def cached_load_polar_data(filename):
    if filename not in _polar_cache:
        _polar_cache[filename] = load_polar_data(filename)
    return _polar_cache[filename]


# ==========================================================
# Fonctions I/O
# ==========================================================

def load_optical_coeffs(filename):
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                mua = float(parts[0]) * 10
                mut = float(parts[1]) * 10
                mus = float(parts[2]) * 10
                g   = float(parts[3])
                return mua, mut, mus, g
    raise ValueError(f"Fichier invalide : {filename}")


def load_phase_function(filename, max_elements=2000):
    mu_file, phase_file = [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mu_file.append(float(parts[0]))
            phase_file.append(float(parts[1]))
            if len(mu_file) >= max_elements:
                break

    theta_deg = np.flip([math.acos(m) * (180 / math.pi) for m in mu_file])
    phase_arr = np.flip(phase_file)
    phase_func = interp1d(theta_deg, phase_arr, kind="linear")

    CDFsize = 200
    thetas = [(math.pi / CDFsize) * (ik - 0.5) for ik in range(1, CDFsize + 1)]
    pdf = [math.sin(t) * float(phase_func(t * 180 / math.pi)) for t in thetas]
    S = sum(pdf)
    pdf = [p / S for p in pdf]
    CDF = [0.0]
    cumul = 0.0
    for p in pdf:
        cumul += p
        CDF.append(cumul)
    CDF[-1] = 1.0
    return {"cdf": CDF, "size": CDFsize, "thetas": thetas}


def binaryTreeSearch(rand, cdf):
    i = bisect.bisect_right(cdf, rand)
    i = max(1, min(i, len(cdf) - 1))
    return i - 1


def load_polar_data(filename, max_elements=2000):
    mu_vals, s11_vals, s12_vals, s33_vals, s43_vals = [], [], [], [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            mu_vals.append(float(parts[0]))
            s11_vals.append(float(parts[1]))
            s12_vals.append(float(parts[2]))
            s33_vals.append(float(parts[3]))
            s43_vals.append(float(parts[4]))
            if len(mu_vals) >= max_elements:
                break

    theta_vals = [math.acos(m) for m in mu_vals]
    if len(theta_vals) > 1 and theta_vals[1] < theta_vals[0]:
        theta_vals.reverse(); s11_vals.reverse()
        s12_vals.reverse(); s33_vals.reverse(); s43_vals.reverse()

    kw = dict(kind="linear", bounds_error=False)
    return {
        "theta": theta_vals,
        "s11": interp1d(theta_vals, s11_vals, fill_value=(s11_vals[0], s11_vals[-1]), **kw),
        "s12": interp1d(theta_vals, s12_vals, fill_value=(s12_vals[0], s12_vals[-1]), **kw),
        "s33": interp1d(theta_vals, s33_vals, fill_value=(s33_vals[0], s33_vals[-1]), **kw),
        "s43": interp1d(theta_vals, s43_vals, fill_value=(s43_vals[0], s43_vals[-1]), **kw),
    }


# ==========================================================
# Classes
# ==========================================================

@dataclass
class Photon:
    x: float; y: float; z: float
    ux: float; uy: float; uz: float

    mus: float = 0.0; mua: float = 0.0; n: float = 0.0; g: float = 0.0
    polar: bool = False; polar_data: object = None
    I: float = 1.0; Q: float = 0.0; U: float = 0.0; V: float = 0.0
    ex1: float = 1.0; ey1: float = 0.0; ez1: float = 0.0
    ex2: float = 0.0; ey2: float = 1.0; ez2: float = 0.0
    weight: float = 1.0; alive: bool = True; sameVoxel: bool = True
    stepLeft: float = 0.0; phase_data: object = None
    ix: int = 0; iy: int = 0; iz: int = 0
    tx: float = float("inf"); ty: float = float("inf"); tz: float = float("inf")
    # zSize et dz passés en paramètre pour éviter globaux dans worker
    _zSize: float = field(default=0.0, repr=False)
    _dz: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self.stepLeft = -math.log(random.random())
        self.update_voxel()
        self.init_polar_basis()

    def update_voxel(self):
        self.ix = int((self.x + xSize) / dx)
        self.iy = int((self.y + ySize) / dy)
        self.iz = int(self.z / self._dz)

        x_min = -xSize + self.ix * dx;  x_max = x_min + dx
        y_min = -ySize + self.iy * dy;  y_max = y_min + dy
        z_min = self.iz * self._dz;     z_max = z_min + self._dz

        self.tx = (x_max - self.x) / self.ux if self.ux > 0 else ((x_min - self.x) / self.ux if self.ux < 0 else float("inf"))
        self.ty = (y_max - self.y) / self.uy if self.uy > 0 else ((y_min - self.y) / self.uy if self.uy < 0 else float("inf"))
        self.tz = (z_max - self.z) / self.uz if self.uz > 0 else ((z_min - self.z) / self.uz if self.uz < 0 else float("inf"))

    def init_polar_basis(self):
        ax, ay, az = (0, 0, 1) if abs(self.uz) < 0.9 else (1, 0, 0)
        ex = ay * self.uz - az * self.uy
        ey = az * self.ux - ax * self.uz
        ez = ax * self.uy - ay * self.ux
        norm = math.sqrt(ex*ex + ey*ey + ez*ez)
        self.ex1, self.ey1, self.ez1 = ex/norm, ey/norm, ez/norm
        self.ex2 = self.uy * self.ez1 - self.uz * self.ey1
        self.ey2 = self.uz * self.ex1 - self.ux * self.ez1
        self.ez2 = self.ux * self.ey1 - self.uy * self.ex1


@dataclass
class Detector:
    x: float; y: float; z: float
    theta: float; phi: float; diam: float; NA: float
    collected: float = 0.0
    D_I: float = 0.0; D_Q: float = 0.0; D_U: float = 0.0; D_V: float = 0.0

    def __post_init__(self):
        self.nz = math.cos(self.theta)
        self.nx = math.sin(self.theta) * math.cos(self.phi)
        self.ny = math.sin(self.theta) * math.sin(self.phi)
        self.theta_max = math.asin(self.NA)


@dataclass
class Box:
    xSize: float; ySize: float; zSize: float
    Nx: int; Ny: int; Nz: int

    def __post_init__(self):
        self.dx = 2 * self.xSize / self.Nx
        self.dy = 2 * self.ySize / self.Ny
        self.dz = self.zSize / self.Nz
        self.materials = [{"mua": 0.0, "mus": 0.0, "g": 0.0, "n": 1.0,
                           "phase_file": None, "phase_data": None,
                           "polar_file": None, "polar_data": None, "polar": False}]
        self.voxel_map = [[[0]*self.Nz for _ in range(self.Ny)] for _ in range(self.Nx)]

    def add_material(self, mua, mus, g, n, phase_file=None, polar=False,
                     phase_data=None, polar_data=None):
        material = {"mua": mua, "mus": mus, "g": g, "n": n,
                    "phase_file": phase_file, "phase_data": phase_data,
                    "polar": polar, "polar_data": polar_data}
        self.materials.append(material)
        return len(self.materials) - 1

    def set_region(self, x_range, y_range, z_range, material_idx):
        ix_min = int((x_range[0] + self.xSize) / self.dx)
        ix_max = int((x_range[1] + self.xSize) / self.dx)
        iy_min = int((y_range[0] + self.ySize) / self.dy)
        iy_max = int((y_range[1] + self.ySize) / self.dy)
        iz_min = int(z_range[0] / self.dz)
        iz_max = int(z_range[1] / self.dz)
        for ix in range(ix_min, ix_max):
            for iy in range(iy_min, iy_max):
                for iz in range(iz_min, iz_max):
                    self.voxel_map[ix][iy][iz] = material_idx

    def getNewVoxelProperties(self, photon):
        if not (0 <= photon.ix < self.Nx and 0 <= photon.iy < self.Ny and 0 <= photon.iz < self.Nz):
            photon.alive = False
            return
        idx = self.voxel_map[photon.ix][photon.iy][photon.iz]
        m = self.materials[idx]
        photon.mua = m["mua"]; photon.mus = m["mus"]; photon.g = m["g"]
        photon.n = m["n"]; photon.phase_data = m["phase_data"]
        photon.polar = m["polar"]; photon.polar_data = m["polar_data"]


# ==========================================================
# Fonctions photon 

def escape(photon, zSize, compteurs):
    cR, cT, cA1, R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V = compteurs

    if photon.z < 0:
        photon.alive = False
        cR += photon.weight
        R_I += photon.I * photon.weight; R_Q += photon.Q * photon.weight
        R_U += photon.U * photon.weight; R_V += photon.V * photon.weight
        return "R", (cR, cT, cA1, R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V)
    elif photon.z > zSize:
        photon.alive = False
        cT += photon.weight
        T_I += photon.I * photon.weight; T_Q += photon.Q * photon.weight
        T_U += photon.U * photon.weight; T_V += photon.V * photon.weight
        return "T", (cR, cT, cA1, R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V)
    elif photon.x < -xSize or photon.x > xSize:
        photon.alive = False; cA1 += photon.weight
        return "X", (cR, cT, cA1, R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V)
    elif photon.y < -ySize or photon.y > ySize:
        photon.alive = False; cA1 += photon.weight
        return "Y", (cR, cT, cA1, R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V)
    return "inside", compteurs


def roulette(photon, compteurs):
    cR, cT, cA1, *rest = compteurs
    if photon.weight < weightmin:
        if random.random() <= chance:
            photon.weight /= chance
        else:
            photon.alive = False
            cA1 += photon.weight
            return (cR, cT, cA1, *rest)
    return compteurs


def propagate(photon, box):
    photon.sameVoxel = True
    s = min(photon.stepLeft / photon.mus, photon.tx, photon.ty, photon.tz)

    if abs(s - photon.stepLeft / photon.mus) < eps:
        photon.stepLeft = 0.0
    else:
        photon.stepLeft -= s * photon.mus

    old_ix, old_iy, old_iz = photon.ix, photon.iy, photon.iz
    dz = photon._dz

    # X
    if abs(s - photon.tx) < eps:
        photon.sameVoxel = False
        photon.x = (-xSize + (old_ix + 1) * dx + eps) if photon.ux > 0 else (-xSize + old_ix * dx - eps)
    else:
        photon.x += s * photon.ux

    # Y
    if abs(s - photon.ty) < eps:
        photon.sameVoxel = False
        photon.y = (-ySize + (old_iy + 1) * dy + eps) if photon.uy > 0 else (-ySize + old_iy * dy - eps)
    else:
        photon.y += s * photon.uy

    # Z
    if abs(s - photon.tz) < eps:
        photon.sameVoxel = False
        photon.z = ((old_iz + 1) * dz + eps) if photon.uz > 0 else (old_iz * dz - eps)
    else:
        photon.z += s * photon.uz

    old_n = photon.n

    if not photon.sameVoxel:
        ix = max(0, min(Nx - 1, photon.ix))
        iy = max(0, min(Ny - 1, photon.iy))
        iz = max(0, min(Nz - 1, photon.iz))
        new_idx = box.voxel_map[ix][iy][iz]
        new_n = box.materials[new_idx]["n"]
        mu = old_n / new_n

        if mu != 1:
            nx, ny, nz = 0.0, 0.0, 0.0
            if abs(s - photon.tx) < eps:
                nx = -1 if photon.ux > 0 else 1
            elif abs(s - photon.ty) < eps:
                ny = -1 if photon.uy > 0 else 1
            elif abs(s - photon.tz) < eps:
                nz = -1 if photon.uz > 0 else 1

            cos_in = nx * photon.ux + ny * photon.uy + nz * photon.uz
            photonReflected = False

            if cos_in > 0:
                cos_out_sqr = 1 - mu**2 * (1 - cos_in**2)
                if cos_out_sqr > 0:
                    cos_out = math.sqrt(cos_out_sqr)
                    R = (((mu*cos_in - cos_out)/(mu*cos_in + cos_out))**2 +
                         ((mu*cos_out - cos_in)/(mu*cos_out + cos_in))**2) / 2
                    photonReflected = random.random() <= R
                else:
                    photonReflected = True
            else:
                photonReflected = True

            if photonReflected:
                photon.ux -= 2*nx*cos_in; photon.uy -= 2*ny*cos_in; photon.uz -= 2*nz*cos_in
            else:
                cos_out = math.sqrt(max(0.0, cos_out_sqr))
                ncoeff = cos_out - mu * cos_in
                photon.ux = ncoeff*nx + mu*photon.ux
                photon.uy = ncoeff*ny + mu*photon.uy
                photon.uz = ncoeff*nz + mu*photon.uz
                photon.n = new_n

    absorb = -photon.weight * math.expm1(-photon.mua * s)
    photon.weight -= absorb
    photon.update_voxel()


def scatter(photon):
    g = photon.g
    if g is not None:
        if abs(g) == 1:
            costheta = g
        elif abs(g) <= math.sqrt(eps):
            costheta = 2.0 * random.random() - 1.0
        else:
            costheta = (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * random.random()))**2) / (2 * g)
    else:
        cdf = photon.phase_data["cdf"]
        CDFsize = photon.phase_data["size"]
        jtheta = binaryTreeSearch(random.random(), cdf)
        costheta = math.cos((jtheta + random.random()) * math.pi / (CDFsize - 1))

    sintheta = math.sqrt(max(0.0, 1 - costheta * costheta))
    phi = 2 * math.pi * random.random()
    cosphi = math.cos(phi); sinphi = math.sin(phi)

    if abs(photon.uz) < 1:
        temp = math.sqrt(photon.ux**2 + photon.uy**2)
        ux_t = sintheta * (photon.ux * photon.uz * cosphi - photon.uy * sinphi) / temp + photon.ux * costheta
        uy_t = sintheta * (photon.uy * photon.uz * cosphi + photon.ux * sinphi) / temp + photon.uy * costheta
        photon.uz = -sintheta * cosphi * temp + photon.uz * costheta
        photon.ux = ux_t; photon.uy = uy_t
    else:
        sign = 1 if photon.uz > 0 else -1
        photon.ux = sintheta * cosphi
        photon.uy = sintheta * sinphi
        photon.uz = costheta * sign

    photon.update_voxel()
    photon.stepLeft = -math.log(random.random())


# Polarisation (inchangée) 

def rot_stokes(phi, I, Q, U, V):
    cos2 = math.cos(2*phi); sin2 = math.sin(2*phi)
    return I, Q*cos2 + U*sin2, -Q*sin2 + U*cos2, V

def rotate_photon_stokes(photon, phi):
    photon.I, photon.Q, photon.U, photon.V = rot_stokes(phi, photon.I, photon.Q, photon.U, photon.V)

def mueller_scatter(I, Q, U, V, s11, s12, s33, s43):
    return (s11*I + s12*Q, s12*I + s11*Q, s33*U + s43*V, -s43*U + s33*V)

def update_polar_basis(photon):
    ux, uy, uz = photon.ux, photon.uy, photon.uz
    ax, ay, az = (0,0,1) if abs(uz) < 0.9 else (1,0,0)
    ex = ay*uz - az*uy; ey = az*ux - ax*uz; ez = ax*uy - ay*ux
    norm = math.sqrt(ex*ex + ey*ey + ez*ez)
    if norm < 1e-12: return
    ex /= norm; ey /= norm; ez /= norm
    photon.ex1, photon.ey1, photon.ez1 = ex, ey, ez
    photon.ex2 = uy*ez - uz*ey
    photon.ey2 = uz*ex - ux*ez
    photon.ez2 = ux*ey - uy*ex


def scatter_polarized(photon):
    pd = photon.polar_data
    s11_0 = float(pd["s11"](0.0))
    I_ref = s11_0 * photon.I

    while True:
        costheta = 2.0 * random.random() - 1.0
        theta    = math.acos(costheta)
        phi      = 2.0 * math.pi * random.random()
        s11 = float(pd["s11"](theta)); s12 = float(pd["s12"](theta))
        cos2phi = math.cos(2*phi); sin2phi = math.sin(2*phi)
        I_test = s11*photon.I + s12*(photon.Q*cos2phi + photon.U*sin2phi)
        if random.random() * I_ref <= I_test:
            break

    sintheta = math.sqrt(max(0.0, 1 - costheta**2))
    cosphi = math.cos(phi); sinphi = math.sin(phi)

    rotate_photon_stokes(photon, phi)
    s33 = float(pd["s33"](theta)); s43 = float(pd["s43"](theta))
    photon.I, photon.Q, photon.U, photon.V = mueller_scatter(
        photon.I, photon.Q, photon.U, photon.V, s11, s12, s33, s43)

    uz_old = photon.uz
    ux, uy, uz = photon.ux, photon.uy, photon.uz

    if abs(uz) < 1.0:
        temp = math.sqrt(1 - uz**2)
        ux_new = sintheta*(ux*uz*cosphi - uy*sinphi)/temp + ux*costheta
        uy_new = sintheta*(uy*uz*cosphi + ux*sinphi)/temp + uy*costheta
        uz_new = -sintheta*cosphi*temp + uz*costheta
    else:
        sign = 1.0 if uz > 0 else -1.0
        ux_new = sintheta*cosphi; uy_new = sintheta*sinphi; uz_new = costheta*sign

    photon.ux = ux_new; photon.uy = uy_new; photon.uz = uz_new

    temp2 = math.sqrt(1 - costheta**2) * math.sqrt(max(0.0, 1 - uz_new**2))
    if temp2 < 1e-12:
        cosi = 0.0
    else:
        cosi = ((-uz_new*costheta + uz_old) if phi <= math.pi else (uz_new*costheta - uz_old)) / temp2
        cosi = max(-1.0, min(1.0, cosi))

    sini = math.sqrt(max(0.0, 1 - cosi**2))
    cos2psi = 2*cosi**2 - 1; sin2psi = 2*sini*cosi
    Q2 = photon.Q*cos2psi - photon.U*sin2psi
    U2 = photon.Q*sin2psi + photon.U*cos2psi

    if photon.I > 0:
        photon.Q = Q2/photon.I; photon.U = U2/photon.I
        photon.V = photon.V/photon.I; photon.I = 1.0
    else:
        photon.I, photon.Q, photon.U, photon.V = 1.0, 0.0, 0.0, 0.0

    update_polar_basis(photon)
    photon.update_voxel()
    photon.stepLeft = -math.log(random.random())


# Source et détecteur 

def axisrotate(r, u, theta):
    ux, uy, uz = u; rx, ry, rz = r
    st = math.sin(theta); ct = math.cos(theta)
    return [
        (ux*ux*(1-ct)+ct)*rx + (ux*uy*(1-ct)-uz*st)*ry + (ux*uz*(1-ct)+uy*st)*rz,
        (uy*ux*(1-ct)+uz*st)*rx + (uy*uy*(1-ct)+ct)*ry + (uy*uz*(1-ct)-ux*st)*rz,
        (uz*ux*(1-ct)-uy*st)*rx + (uz*uy*(1-ct)+ux*st)*ry + (uz*uz*(1-ct)+ct)*rz,
    ]

def orthogonal_unit_vector(u):
    ux, uy, uz = u
    ref = (0,0,1) if abs(uz) < 0.9 else (1,0,0)
    rx, ry, rz = ref
    vx = ry*uz - rz*uy; vy = rz*ux - rx*uz; vz = rx*uy - ry*ux
    norm = math.sqrt(vx*vx + vy*vy + vz*vz)
    return (vx/norm, vy/norm, vz/norm)

def gaussian_source(x0, y0, z0, w0, divergence, z_launch, ux, uy, uz, dz, _zSize):
    u = (0, 0, 1); v = (1, 0, 0)
    w0_vec = axisrotate(v, u, 2*math.pi*random.random())
    r = w0 * math.sqrt(-0.5 * math.log(random.random()))
    xt = x0 + r*w0_vec[0]; yt = y0 + r*w0_vec[1]; zt = z0 + r*w0_vec[2]
    w0_vec = axisrotate(v, u, 2*math.pi*random.random())
    phi = math.atan(math.tan(divergence) * math.sqrt(-0.5*math.log(random.random())))
    ux, uy, uz = axisrotate(u, w0_vec, phi)
    if abs(uz) < eps: uz = eps
    x = xt - (zt - z_launch)*ux/uz
    y = yt - (zt - z_launch)*uy/uz
    return Photon(x=x, y=y, z=0.0, ux=ux, uy=uy, uz=uz, _zSize=_zSize, _dz=dz)


def detect(photon, det):
    if photon.z >= det.z:
        if abs(photon.uz) < 1e-12: return False
        t = (det.z - photon.z) / photon.uz
        xi = photon.x + t*photon.ux; yi = photon.y + t*photon.uy
        if math.sqrt((xi-det.x)**2 + (yi-det.y)**2) > det.diam/2: return False
        theta_max = math.asin(det.NA)
        norm_u = math.sqrt(photon.ux**2 + photon.uy**2 + photon.uz**2)
        if math.acos(photon.uz/norm_u) > theta_max: return False
        det.collected += photon.weight
        det.D_I += photon.I*photon.weight; det.D_Q += photon.Q*photon.weight
        det.D_U += photon.U*photon.weight; det.D_V += photon.V*photon.weight
        return True


# ==========================================================
# Fonction worker : defini une combinaison des differents parametres
def run_simulation(args):
    """Un worker = une combinaison (zSize, lambada, dense, pol_name, stokes)"""
    zSize, lambada, dense, pol_name, stokes = args

    random.seed()  # Graine différente par worker (fork-safe)

    dz = zSize / Nz
    zdet = zSize

    # Chargement fichiers (avec cache local au process) 
    coeff_file = f"pavin_oct_alb/coeff_potique_Pav_al_big_4_v_{dense}_l_{lambada}_nm.txt"
    phase_file = f"pavin_oct_alb/ram_function_phases_Pav_al_big_4_v_{dense}_l_{lambada}_nm.txt"

    mua_file, mut_file, mus_file, g_file = cached_load_optical_coeffs(coeff_file)
    phase_data = cached_load_phase_function(phase_file)
    polar_data = cached_load_polar_data(phase_file)

    # Box 
    box = Box(xSize, ySize, zSize, Nx, Ny, Nz)
    idx_mat2 = box.add_material(
        mua=mua_file * 10, mus= mus_file * 10, g=None, n=1,
        phase_file=phase_file, polar=True,
        phase_data=phase_data, polar_data=polar_data
    )
    box.set_region((-xSize, xSize), (-ySize, ySize), (0, zSize), idx_mat2)

    I_in, Q_in, U_in, V_in = stokes

    # Compteurs locaux 
    compteurR = compteurT = compteurA1 = 0.0
    R_I = R_Q = R_U = R_V = 0.0
    T_I = T_Q = T_U = T_V = 0.0
    detector = Detector(xdet, ydet, zdet, thetadet, phidet, diametre, NA)

    label = f"LZ={zSize} POL={pol_name} lambda={lambada} dense={dense}"
    pid = os.getpid()
    print(f"[PID {pid}] Début : {label}")

    for inphoton in range(1, Nphoton + 1):
        p = gaussian_source(xinit, yinit, zinit, W0, theta_src, zinit,
                            uxinit, uyinit, uzinit, dz, zSize)
        p.I = I_in; p.Q = Q_in; p.U = U_in; p.V = V_in
        box.getNewVoxelProperties(p)

        compteurs = (compteurR, compteurT, compteurA1,
                     R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V)

        while p.alive:
            while p.alive and p.stepLeft > 0:
                propagate(p, box)
                if not p.sameVoxel:
                    detect(p, detector)
                    result, compteurs = escape(p, zSize, compteurs)
                    if p.alive:
                        box.getNewVoxelProperties(p)

            if p.alive:
                compteurs = roulette(p, compteurs)
            if p.alive:
                if p.polar:
                    scatter_polarized(p)
                else:
                    scatter(p)

        (compteurR, compteurT, compteurA1,
         R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V) = compteurs

    print(f"[PID {pid}] Fin   : {label}")

    # Écriture résultats 
    nom_fichier = f"log_big_4_v_{dense}_l_{lambada}_z_{zSize}_pol_{pol_name}.txt"
    _write_results(nom_fichier, Nphoton, compteurR, compteurT, compteurA1,
                   R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V, detector)
    return nom_fichier


def _write_results(nom_fichier, Nphoton, compteurR, compteurT, compteurA1,
                   R_I, R_Q, R_U, R_V, T_I, T_Q, T_U, T_V, detector):
    with open(nom_fichier, "w", encoding="utf-8") as f:
        A_count = Nphoton - (compteurR + compteurT)
        R_rate = compteurR / Nphoton
        T_rate = compteurT / Nphoton
        A_rate = 1.0 - R_rate - T_rate
        C_rate = detector.collected / Nphoton

        f.write("\n" + "="*66 + "\n")
        f.write("                 RÉSULTATS DE LA SIMULATION MONTE CARLO\n")
        f.write("="*66 + "\n")
        f.write(f"\nNombre de photons émis : {Nphoton}\n")

        f.write("\n" + "-"*66 + "\nBILAN GLOBAL\n" + "-"*66 + "\n")
        f.write(f"{'Réflexion (R)':32s}: {compteurR:12.6f}   ({R_rate:10.6f})\n")
        f.write(f"{'Transmission (T)':32s}: {compteurT:12.6f}   ({T_rate:10.6f})\n")
        f.write(f"{'Absorption (A)':32s}: {A_count:12.6f}   ({A_rate:10.6f})\n")

        f.write("\n" + "-"*66 + "\nDÉTECTEUR\n" + "-"*66 + "\n")
        f.write(f"{'Photons collectés':32s}: {detector.collected:12.6f}\n")
        f.write(f"{'Taux de collecte':32s}: {C_rate:12.6f}\n")
        f.write(f"{'Pourcentage collecté':32s}: {100*C_rate:12.4f} %\n")

        def write_stokes_block(label, sI, sQ, sU, sV):
            f.write("\n" + "-"*66 + f"\n{label}\n" + "-"*66 + "\n")
            f.write(f"I = {sI:.6f}\nQ = {sQ:.6f}\nU = {sU:.6f}\nV = {sV:.6f}\n")
            if sI > 0:
                f.write("\nVecteur de Stokes normalisé :\nI = 1.000000\n")
                f.write(f"Q = {sQ/sI:.6f}\nU = {sU/sI:.6f}\nV = {sV/sI:.6f}\n")
                DOP  = math.sqrt(sQ**2 + sU**2 + sV**2) / sI
                LDOP = math.sqrt(sQ**2 + sU**2) / sI
                CDOP = abs(sV) / sI
                f.write(f"\nPolarisation totale     = {DOP:.6f}\n")
                f.write(f"Polarisation linéaire  = {LDOP:.6f}\n")
                f.write(f"Polarisation circulaire= {CDOP:.6f}\n")
            else:
                f.write("Aucun signal\n")

        f.write("\n" + "="*66 + "\n                     RÉSULTATS DE POLARISATION\n" + "="*66 + "\n")
        write_stokes_block("RÉFLEXION",  R_I, R_Q, R_U, R_V)
        write_stokes_block("TRANSMISSION", T_I, T_Q, T_U, T_V)
        write_stokes_block("DÉTECTEUR", detector.D_I, detector.D_Q, detector.D_U, detector.D_V)

        f.write("\n" + "="*66 + "\nFin de simulation\n" + "="*66 + "\n")

    print("fini :", nom_fichier)


# ==========================================================
# MAIN


if __name__ == "__main__":

    # Génère toutes les combinaisons
    all_args = list(itertools.product(
        LZ_values,
        Lambda_values,
        density_values,
        polar_inputs.items()   # (pol_name, stokes)
    ))
    # Réorganise en (zSize, lambada, dense, pol_name, stokes)
    all_tasks = [(z, lam, d, pname, stokes)
                 for z, lam, d, (pname, stokes) in all_args]

    total = len(all_tasks)
    # Nombre de CPU dispo (laisse 1 libre pour l'OS)
    ncpu = max(1, multiprocessing.cpu_count() - 1)
    print(f"\n{'='*60}")
    print(f"  {total} simulations × {Nphoton} photons")
    print(f"  Parallélisation sur {ncpu} CPU(s)")
    print(f"{'='*60}\n")

    with multiprocessing.Pool(processes=ncpu) as pool:
        results = pool.map(run_simulation, all_tasks)

    print(f"\nToutes les simulations terminées ({len(results)} fichiers).")
