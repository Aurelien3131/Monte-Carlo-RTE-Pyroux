# ======================================================================================================================

# polarization.py

# Polarized photon scattering management.

# Contains:
# - Stokes vector rotation
# - Mueller matrix application
# - photon direction update after scattering
# - update of local polarization basis
# - angular sampling taking polarization state into account


# This module is called during scattering events
# when polarized mode is enabled.

# ======================================================================================================================


import math
import random


def rot_stokes(phi, I, Q, U, V):

    cos2 = math.cos(2.0 * phi)
    sin2 = math.sin(2.0 * phi)

    I2 = I
    Q2 = Q * cos2 + U * sin2
    U2 = -Q * sin2 + U * cos2
    V2 = V

    return I2, Q2, U2, V2

def rotate_photon_stokes(photon, phi):
    photon.I, photon.Q, photon.U, photon.V = rot_stokes(phi,photon.I,photon.Q,photon.U,photon.V)



def mueller_scatter(I, Q, U, V, s11, s12, s33, s43):

    I2 = s11 * I + s12 * Q
    Q2 = s12 * I + s11 * Q
    U2 = s33 * U + s43 * V
    V2 = -s43 * U + s33 * V

    return I2, Q2, U2, V2
    
    
    
   
    
def update_polar_basis(photon): # Updates the photon's local rotation vector

    ux, uy, uz = photon.ux, photon.uy, photon.uz

    # Choose a reference vector
    if abs(uz) < 0.9:
        ax, ay, az = 0.0, 0.0, 1.0
    else:
        ax, ay, az = 1.0, 0.0, 0.0

    # e1 = ax × u
    ex = ay*uz - az*uy
    ey = az*ux - ax*uz
    ez = ax*uy - ay*ux

    norm = math.sqrt(ex*ex + ey*ey + ez*ez)
    if norm < 1e-12:
        return

    ex /= norm
    ey /= norm
    ez /= norm

    # e2 = u × e1
    ex2 = uy*ez - uz*ey
    ey2 = uz*ex - ux*ez
    ez2 = ux*ey - uy*ex

    photon.ex1, photon.ey1, photon.ez1 = ex, ey, ez
    photon.ex2, photon.ey2, photon.ez2 = ex2, ey2, ez2
    

def scatter_polarized(photon):

    pd = photon.polar_data
    s11_0 = float(pd["s11"](0.0))
    I_ref  = s11_0 * photon.I # Reference maximum at θ=0

    while True:
        # Uniform sampling of scattering angles in cos(theta) — like Ramella-code acos(2·rnd−1)
        costheta = 2.0 * random.random() - 1.0
        theta    = math.acos(costheta)
        phi      = 2.0 * math.pi * random.random()

        s11 = float(pd["s11"](theta))
        s12 = float(pd["s12"](theta))

        cos2phi = math.cos(2.0 * phi)
        sin2phi = math.sin(2.0 * phi)

        # Comprehensive criterion including current polarization
        I_test = s11 * photon.I + s12 * (photon.Q * cos2phi + photon.U * sin2phi)

        if random.random() * I_ref <= I_test:
            break

    sintheta = math.sqrt(max(0.0, 1.0 - costheta * costheta))
    cosphi   = math.cos(phi)
    sinphi   = math.sin(phi)

    # ROTATION 1 : Stokes → Scattering plan
    rotate_photon_stokes(photon, phi)   # Unchangé

    # MUELLER
    s33 = float(pd["s33"](theta))
    s43 = float(pd["s43"](theta))

    photon.I, photon.Q, photon.U, photon.V = mueller_scatter(
        photon.I, photon.Q, photon.U, photon.V, s11, s12, s33, s43
    )

    # DIRECTION UPDATE (uz_old required for rotation 2)
    uz_old = photon.uz
    ux, uy, uz = photon.ux, photon.uy, photon.uz

    if abs(uz) < 1.0:
        temp   = math.sqrt(1.0 - uz * uz)
        ux_new = sintheta * (ux * uz * cosphi - uy * sinphi) / temp + ux * costheta
        uy_new = sintheta * (uy * uz * cosphi + ux * sinphi) / temp + uy * costheta
        uz_new = -sintheta * cosphi * temp + uz * costheta
    else:
        sign   = 1.0 if uz > 0 else -1.0
        ux_new = sintheta * cosphi
        uy_new = sintheta * sinphi
        uz_new = costheta * sign

    photon.ux = ux_new
    photon.uy = uy_new
    photon.uz = uz_new

    temp2 = math.sqrt(1.0 - costheta**2) * math.sqrt(max(0.0, 1.0 - uz_new**2))

    if temp2 < 1e-12:
        cosi = 0.0
    else:
        if phi > math.pi:
            cosi = ( uz_new * costheta - uz_old) / temp2
        else:
            cosi = (-uz_new * costheta + uz_old) / temp2
        cosi = max(-1.0, min(1.0, cosi))

    sini    = math.sqrt(max(0.0, 1.0 - cosi * cosi))
    cos2psi =  2.0 * cosi**2 - 1.0
    sin2psi =  2.0 * sini * cosi

    Q2 = photon.Q * cos2psi - photon.U * sin2psi
    U2 = photon.Q * sin2psi + photon.U * cos2psi

    if photon.I > 0:
        photon.Q = Q2 / photon.I
        photon.U = U2 / photon.I
        photon.V = photon.V / photon.I
        photon.I = 1.0
    else:
        photon.I, photon.Q, photon.U, photon.V = 1.0, 0.0, 0.0, 0.0

    update_polar_basis(photon)

    # New free step
    photon.update_voxel()
    photon.stepLeft = -math.log(random.random())

 
 
