# ======================================================================================================================

# propagation.py

# Photon propagation in a scattering medium.

# Contains:
# - photon displacement inside voxels
# - optical interface handling
# - boundary conditions (reflection / transmission)
# - Snell-Descartes law and Fresnel coefficients
# - absorption in the medium
# - Russian roulette for statistical termination


# This module updates the position, time, weight
# and state of the photon after each propagation step.

# ======================================================================================================================

import math
import random
import numpy as np
from config import *




def escape(photon, xSize, ySize, zSize):
    """
    Returns:
    - "R" : reflection
    - "T" : transmission
    - "X" : lateral x exit
    - "Y" : lateral y exit
    - "inside" : still inside the medium
    """

    # réflexion
    if photon.z < 0:
        photon.alive = False
        return "R"

    # transmission
    elif photon.z > zSize:
        photon.alive = False
        return "T"

    # sortie x
    elif photon.x < -xSize or photon.x > xSize:
        photon.alive = False
        return "X"

    # sortie y
    elif photon.y < -ySize or photon.y > ySize:
        photon.alive = False
        return "Y"

    return "inside"

def escape_periodic_xy(photon, zSize, xSize, ySize):
    """
    reflection
    periodic boundaries in x and y:
    - z < 0      => reflection
    - z > zSize  => transmission
    - x, y       => periodic wrap
    """

    # réflexion
    if photon.z < 0:
        photon.alive = False
        return "R"

    # transmission
    elif photon.z > zSize:
        photon.alive = False
        return "T"

    # périodique en x
    if photon.x > xSize:
        photon.x -= 2 * xSize
    elif photon.x < -xSize:
        photon.x += 2 * xSize

    # périodique en y
    if photon.y > ySize:
        photon.y -= 2 * ySize
    elif photon.y < -ySize:
        photon.y += 2 * ySize

    return "inside"
    
def roulette(photon, weightmin, chance):
    """
    Russian roulette:
    if weight too small, photon either survives or dies
    """

    if photon.weight < weightmin:

        rn = random.random()

        if rn <= chance:
            photon.weight /= chance
            return "survive"
        else:
            photon.alive = False
            return "dead"

    return "ok"

def propagate(photon, box):

	photon.sameVoxel = True;
	
	s = min(photon.stepLeft/photon.mus, min(photon.tx, min(photon.ty ,photon.tz )) )
	dz = photon.dz
        
	if (s == photon.stepLeft/photon.mus):
		photon.stepLeft = 0.0
	else :
		photon.stepLeft -= s * photon.mus


	# 1. Propagation
	old_ix = photon.ix
	old_iy = photon.iy
	old_iz = photon.iz
	
	old_pos_x = photon.x
	old_pos_y = photon.y	
	old_pos_z = photon.z
		
	# X
	if  (abs(s - photon.tx) < eps):
		photon.sameVoxel = False
		if (photon.ux > 0 ):
			photon.x = -xSize + (old_ix + 1) * dx + eps
		else:
			photon.x = -xSize + old_ix *dx - eps
	else:
		photon.x += s*photon.ux
		 
	
	# Y
	if (abs(s - photon.ty) < eps):
		photon.sameVoxel = False
		if (photon.uy > 0 ):
			photon.y = -ySize + (old_iy + 1) * dy + eps
		else:
			photon.y = -ySize + old_iy *dy - eps
	else:
		photon.y += s*photon.uy
		
			
	# Z
	if (abs(s - photon.tz) < eps):
		photon.sameVoxel = False
		if (photon.uz > 0 ):
			photon.z = (old_iz + 1) * dz + eps
		else:
			photon.z = old_iz *dz - eps
	else:
		photon.z += s*photon.uz
		

	old_n = photon.n
	
	
	
	distance = np.sqrt((old_pos_x - photon.x)**2 + (old_pos_y - photon.y)**2 + (old_pos_z - photon.z)**2)	
	time = distance*old_n/(C_VELOCITY*100) # distances in cm and C_VELOCITY en m/s so need a 100 factor
	
	# 2. Snell-Descartes / Fresnel
	if photon.sameVoxel == False :
	 
		
		# new voxel
		ix = max(0, min(Nx - 1, photon.ix))
		iy = max(0, min(Ny - 1, photon.iy))
		iz = max(0, min(Nz - 1, photon.iz))
		# propreties of the new voxel
		new_idx = box.voxel_map[ix][iy][iz]
		new_n = box.materials[new_idx]["n"]
		mu = old_n / new_n
		
		# 2 cases of mu : 
		# mu!=1:
		if mu != 1:  # refractive indices are different, so refraction/reflection laws must be applied
			photonReflected = False

		
			nx, ny, nz = 0.0, 0.0, 0.0

			if abs(s - photon.tx) < eps:
				nx = -1 if photon.ux > 0 else 1
			elif abs(s - photon.ty) < eps:
				ny = -1 if photon.uy > 0 else 1
			elif abs(s - photon.tz) < eps:
				nz = -1 if photon.uz > 0 else 1

			# scalar product
			cos_in = nx*photon.ux + ny*photon.uy + nz*photon.uz

			if cos_in > 0:
				
				cos_out_sqr = 1 - mu**2 * (1 - cos_in**2)

				if cos_out_sqr > 0:

					cos_out = math.sqrt(cos_out_sqr)
					R = (((mu*cos_in - cos_out)/(mu*cos_in + cos_out))**2 + ((mu*cos_out - cos_in)/(mu*cos_out + cos_in))**2) / 2
					photonReflected = random.random() <= R

			else:
				photonReflected = True  # Total reflexion

			if photonReflected:
				# reflexion
				photon.ux -= 2*nx*cos_in
				photon.uy -= 2*ny*cos_in
				photon.uz -= 2*nz*cos_in

			else:
				# refraction
				cos_out = math.sqrt(max(0.0, cos_out_sqr))
				ncoeff = cos_out - mu*cos_in

				photon.ux = ncoeff*nx + mu*photon.ux
				photon.uy = ncoeff*ny + mu*photon.uy
				photon.uz = ncoeff*nz + mu*photon.uz
				photon.n = new_n   # update refractive index
		
		# mu == 1:
		if mu == 1: # Same refractive index -> nothing is done
			pass

	
	photon.timephot +=time
	 
	# 3. Absorption
	absorb = -photon.weight * math.expm1(-photon.mua * s)
	photon.weight -= absorb
	
    
	 # Recompute voxel position and distances
	photon.update_voxel()
	
