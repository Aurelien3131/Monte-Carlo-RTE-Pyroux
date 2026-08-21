# ======================================================================================================================

# sources.py

# Generation and initialization of source photons.

# Contains:
# - point sources and plane waves
# - Gaussian beam generation
# - temporal modulation of the emission
# - spatial and angular sampling
# - photon initialization before propagation

# Convention:
# emission from the z = 0 plane
# initial propagation along the +Z direction

# ======================================================================================================================

import math
import random

from config import *
from photon import Photon



# Photon source functions

def axisrotate(r, u, theta):
	"""
	Rotation of vector r around the unit axis u by an angle theta.
	(Copied from MCmatlab)
	"""
	ux, uy, uz = u
	rx, ry, rz = r
	
	st = math.sin(theta)
	ct = math.cos(theta)
	out = [0.0, 0.0, 0.0]
	
	out[0] = (ux*ux*(1-ct) + ct)*rx + (ux*uy*(1-ct) - uz*st)*ry + (ux*uz*(1-ct) + uy*st)*rz
	out[1] = (uy*ux*(1-ct) + uz*st)*rx + (uy*uy*(1-ct) + ct)*ry + (uy*uz*(1-ct) - ux*st)*rz
	out[2] = (uz*ux*(1-ct) - uy*st)*rx + (uz*uy*(1-ct) + ux*st)*ry + (uz*uz*(1-ct) + ct)*rz

	return out

def orthogonal_unit_vector(u):
    ux, uy, uz = u

    # Choose a non-parallel vector
    if abs(uz) < 0.9:
        ref = (0.0, 0.0, 1.0)
    else:
        ref = (1.0, 0.0, 0.0)

    rx, ry, rz = ref

    # Cross product ref × u
    vx = ry*uz - rz*uy
    vy = rz*ux - rx*uz
    vz = rx*uy - ry*ux

    norm = math.sqrt(vx*vx + vy*vy + vz*vz)

    return (vx/norm, vy/norm, vz/norm)

def isotrope(x0, y0, z0, dz, w0, divergence, z_launch, ux, uy, uz, freq, nb_period, fact_modu):

    # Emission time
    t_emit = 0

    # Modulation
    weight = 1.0 + fact_modu * math.cos(2.0 * math.pi * freq * t_emit)

    # Initial position
    x = x0
    y = y0
    z = z0

    # Uniform isotropic direction over the sphere
    mu = 2.0 * random.random() - 1.0
    phi = 2.0 * math.pi * random.random()

    sin_theta = math.sqrt(1.0 - mu * mu)

    ux = sin_theta * math.cos(phi)
    uy = sin_theta * math.sin(phi)
    uz = mu

    return Photon( x=x, y=y, z=z, dz=dz, ux=ux, uy=uy, uz=uz, weight=weight, t_emit=t_emit, freq=freq )

def gaussian_source(x0, y0, z0, dz, w0, divergence, z_launch, ux, uy, uz, freq, nb_period, fact_modu):

    # Emission time
    t_emit = 0

    # Modulation
    weight = 1.0 + fact_modu * math.cos(2.0 * math.pi * freq * t_emit)

    u = (0.0, 0.0, 1.0)   # Laser propagation direction
    v = (1.0, 0.0, 0.0)   # Vector orthogonal to the laser beam

    # ==========================================================
    # 1. Target point in the beam waist: NEAR FIELD
    # ==========================================================

    w0_vec = axisrotate(v, u, 2.0 * math.pi * random.random() )

    r = w0 * math.sqrt(-0.5 * math.log(random.random())    )

    xt = x0 + r * w0_vec[0]
    yt = y0 + r * w0_vec[1]
    zt = z0 + r * w0_vec[2]

    # 2. Beam direction: FAR FIELD
    w0_vec = axisrotate(v,u,2.0 * math.pi * random.random()    )

    phi = math.atan(math.tan(divergence)* math.sqrt(-0.5 * math.log(random.random()))    )

    ux, uy, uz = axisrotate(u, w0_vec, phi)


    # 3. Projection onto the z = z_launch plane
    if abs(uz) < eps:
        uz = eps

    x = xt - (zt - z_launch) * ux / uz
    y = yt - (zt - z_launch) * uy / uz
    z = z_launch


    # 4. Photon initialization
    return Photon( x=x, y=y, z=z, dz=dz, ux=ux, uy=uy, uz=uz, weight=weight, t_emit=t_emit, freq=freq )
	
	
	
def plane_wave(x0, y0, z0, dz, w0, divergence, z_launch, ux, uy, uz, freq, nb_period, fact_modu):


    # 1. Emission time
    t_emit = 0


    # 2. Modulation
    weight = 1.0 + fact_modu * math.cos(2.0 * math.pi * freq * t_emit    )


    # 3. Uniform position over the source surface
    x = (random.random() - 0.5) * 2.0 * xSize
    y = (random.random() - 0.5) * 2.0 * ySize
    z = z_launch


    # 4. Fixed propagation direction
    ux = uxinit
    uy = uyinit
    uz = uzinit

    # Normalize direction
    norm = math.sqrt(ux * ux + uy * uy + uz * uz )
    ux /= norm
    uy /= norm
    uz /= norm


    # 5. Photon initialization


    return Photon(x=x, y=y, z=z, dz=dz ,ux=ux ,uy=uy ,uz=uz ,weight=weight ,t_emit=t_emit ,freq=freq )



def gaussian_source_modu(x0, y0, z0, dz, w0, divergence,z_launch, ux, uy, uz, freq, nb_period, fact_modu):

	# Emission time (continuous or photon-index based)
	t_emit = random.random() * nb_period / freq   # Uniform over one period

	# Modulation
	weight = 1.0 + fact_modu * math.cos(2 * math.pi * freq * t_emit)

	u = (0 ,0 ,1) # Laser propagation direction vector
	v = (1 ,0 ,0) # Vector orthogonal to the laser beam

	# 1. Target point in the beam waist: NEAR FIELD
	w0_vec = axisrotate(v, u, 2.0 * math.pi * random.random())
	r = w0 * math.sqrt(-0.5 * math.log(random.random()))

	xt = x0 + r * w0_vec[0]
	yt = y0 + r * w0_vec[1]
	zt = z0 + r * w0_vec[2]

	# 2. Beam direction (divergence): FAR FIELD
	w0_vec = axisrotate(v, u, 2.0 * math.pi * random.random())
	phi = math.atan(math.tan(divergence) * math.sqrt(-0.5 * math.log(random.random())))

	ux, uy, uz = axisrotate(u, w0_vec, phi)

	# 3. Projection onto the z = z_launch plane
	if abs(uz) < eps: # Numerical safety
		uz = eps
	x = xt - (zt - z_launch) * ux / uz
	y = yt - (zt - z_launch) * uy / uz
	z = 0 # z_launch plane

	return Photon(x=x, y=y, z=z, dz=dz, ux=ux, uy=uy, uz=uz, weight=weight, t_emit=t_emit, freq=freq)

