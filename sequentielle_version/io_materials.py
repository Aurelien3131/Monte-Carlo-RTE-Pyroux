# ======================================================================================================================

# io_materials.py

# Reading and preprocessing of the optical properties of the medium.

# Contains:
# - loading of optical coefficients (mua, mus, mut, g)
# - reading phase functions
# - construction of CDFs for Monte Carlo sampling
# - loading polarized scattering data (Mueller matrices)
# - interpolation of tabulated data

# The loaded data are then used by:
# geometry.py      → assignment of optical properties to voxels
# scattering.py    → computation of scattering events and update of the photon's direction vectors (ux, uy, uz)
# polarization.py  → computation of polarized scattering events and update of the photon's direction vectors (ux, uy, uz)

# ======================================================================================================================



import math
import bisect

import numpy as np
from scipy.interpolate import interp1d


# Fonction chargement fonction de phase et contruction cdf
def load_phase_function(filename, max_elements=2000):
    """
    File format:
        cos(theta)   phase(theta)

    return :
        {
            "cdf": [...],     # taille CDFSIZE+1
            "size": CDFSIZE
        }
    """

    mu_file = []
    phase_file = []

    # 1.  Read the file
    with open(filename, "r") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            mu = float(parts[0])
            val = float(parts[1])

            mu_file.append(mu)
            phase_file.append(val)

            if len(mu_file) >= max_elements:
                break

    if len(mu_file) < 2:
        raise ValueError("Phase function file needs at least 2 rows")
        
    # 2. Construct the continuous phase function for the CDF
    theta_deg = []
    for ij in range(0,len(mu_file)):
        theta_deg.append( math.acos(mu_file[ij]) * (180/math.pi) )
    
    theta_deg = np.flip(theta_deg)
    phase_file = np.flip(phase_file)
    phase_func = interp1d(theta_deg, phase_file, kind='linear')



    # 3. Construct the CDF
    CDF_SIZE = 200 # number of CDF elements, defined by mcmatlab
    thetas =[]
    for ik in range (1,CDF_SIZE+1):
        thetas.append((math.pi/CDF_SIZE)*(ik - 0.5))
    pdf = []
    for il in range(0, CDF_SIZE):
        pdf.append(math.sin(thetas[il]) * phase_func(thetas[il] * 180/math.pi))
    Stotal=sum(pdf)
    if Stotal <= 0:
        raise ValueError("PDF invalid (sum <= 0)")
    
    pdf = [p / Stotal for p in pdf]
    
    CDF = [0.0]
    cumul = 0.0
    for p in pdf:
        cumul += p
        CDF.append(cumul)

    CDF[-1] = 1.0  # Security



    return {"cdf": CDF, "size": CDF_SIZE, "thetas": thetas}

def binaryTreeSearch(rand, cdf):
    # bisect_right finds i such that cdf[i-1] <= rand < cdf[i]
    # we want j such that cdf[j] < rand <= cdf[j+1], therefore j = i-1
    i = bisect.bisect_right(cdf, rand)
    i = max(1, min(i, len(cdf) - 1))
    return i - 1

def load_polar_data(filename, max_elements=2000):

    mu_vals  = []
    s11_vals = []
    s12_vals = []
    s33_vals = []
    s43_vals = []

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

    if len(mu_vals) < 2:
        raise ValueError("Polar file too short for interpolation")

    # Increasing theta values
    theta_vals = [math.acos(mu) for mu in mu_vals]

    # If mu goes from +1 to -1 then theta is already increasing
    # Otherwise reverse the arrays
    if theta_vals[1] < theta_vals[0]:

        theta_vals.reverse()
        s11_vals.reverse()
        s12_vals.reverse()
        s33_vals.reverse()
        s43_vals.reverse()

  # For normalization

    
    data = {
        "theta": theta_vals,
        "s11": interp1d(theta_vals, s11_vals, kind='linear',
                        bounds_error=False,
                        fill_value=(s11_vals[0], s11_vals[-1])),

        "s12": interp1d(theta_vals, s12_vals, kind='linear',
                        bounds_error=False,
                        fill_value=(s12_vals[0], s12_vals[-1])),

        "s33": interp1d(theta_vals, s33_vals, kind='linear',
                        bounds_error=False,
                        fill_value=(s33_vals[0], s33_vals[-1])),

        "s43": interp1d(theta_vals, s43_vals, kind='linear',
                        bounds_error=False,
                        fill_value=(s43_vals[0], s43_vals[-1]))
    }

    return data
def load_optical_coeffs(filename):
    """
    Returns mua, mus, mut, and g from the file.
    mua and mus are multiplied by 10.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # garder seulement lignes utiles
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        data_lines.append(line)

    if len(data_lines) < 1:
        raise ValueError(f"No data lines found in {filename}")

    parts = data_lines[0].split()

    if len(parts) < 4:
        raise ValueError(f"The file {filename} must contain 4 columns: mua mut mus g")

    mua = float(parts[0]) * 10 # en cm^-1
    mut = float(parts[1]) * 10 # en cm^-1
    mus = float(parts[2]) * 10 # en cm^-1
    g   = float(parts[3])

    return mua, mut, mus, g   

