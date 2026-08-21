# ======================================================================================================================

# config.py

# Global parameters for the Monte Carlo simulation.

# Contains:
# - physical constants
# - Monte Carlo parameters
# - simulation geometry
# - source / detector parameters
# - histogram configuration
# - temporal modulation
# - parametric loops (wavelength, density, polarization, thickness)

# Convention:
# Source    → z = 0
# Detector  → z = zSize
# Main propagation along the +Z direction

# This module is imported by all simulation modules.

# ======================================================================================================================


import math


# ==========================================================
# CPU parallelization
# ==========================================================


NPROC = 10



# ==========================================================
# Physical constants
# ==========================================================

C_VELOCITY = 299792458      # m/s

# ==========================================================
# Simulation loops
# ==========================================================

polar_inputs = {
    "1000": (1, 0, 0, 0),
    #"1001": (1,0,0,1),
    #"1010": (1,0,1,0),
    #"1100": (1,1,0,0),
}

density_values = [
    40,
    #35,
    #47,
    #65,
    #124,
]

Lambda_values = [
    9.8,
    #0.55,
    #1.55,
    #4,
]

LZ_values = [1,
    #0,
    #500,
    #1000,
    #2000,
    #3000,
    #4000,
]

# ==========================================================
# Monte Carlo
# ==========================================================

Nphoton = 100_000_00

weightmin = 0.001
chance = 0.1
eps = 1e-10

# ==========================================================
# Geometry
# ==========================================================

xSize = 100.0      # cm
ySize = 100.0      # cm

Nx = 100
Ny = 100
Nz = 100

dx = 2 * xSize / Nx
dy = 2 * ySize / Ny

# dz depends on zSize
# it will be recomputed in the main loop

boundary_function = 1 # choose the boundary conditions
# = 1 -> all boundaries are escaping
# = 2 -> x/y boundaries are periodic

# ==========================================================
# Source
# ==========================================================

source_function = 1 # choose the light source
# = 0 -> pencil beam
# = 1 -> gaussian source modulated
# = 2 -> isotrope
# = 3 -> gaussian_source
# = 4 -> plane_wave

xinit = 0.0
yinit = 0.0
zinit = 0.0

zlaunch = zinit

uxinit = 0.0
uyinit = 0.0
uzinit = 1.0

# Gaussian source
def laser_params(lambda_value):  # retrieves the laser w0 and theta values corresponding to the given wavelength
    tol = 1e-6
    fichier = "param_laser.txt"
    with open(fichier, "r") as f:
        for ligne in f:
            if ligne.startswith("#") or not ligne.strip():
                continue
            lambda_fichier, w0, theta = map(float, ligne.split())

            if abs(lambda_fichier - lambda_value) < tol:
                return w0, theta

    raise ValueError(f"Lambda = {lambda_value} not found")


# ==========================================================
# Detector
# ==========================================================

xdet = 0.0
ydet = 0.0
#zdet = 0  # defined in the main program

thetadet = math.pi  # detector oriented toward -z
#thetadet = 0  # detector oriented toward +z
phidet = 0.0  # azimuthal angle in the XY plane

diametre = 15.0
NA = 1

# ==========================================================
# Histograms
# ==========================================================

bins = 4000

# ==========================================================
# Modulation
# ==========================================================

frequency = [1e3,1e4,1e5,1e6,1e7,1e8]
nb_period = 2
modulation = [1]

#t_ballistic = zdet
#time_window = (nb_period/freq)+0.1e-7
time_gate = 0.1e-7
