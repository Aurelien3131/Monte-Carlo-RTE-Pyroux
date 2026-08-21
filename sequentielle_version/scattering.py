# ======================================================================================================================

# scattering.py

# Management of unpolarized photon scattering.

# Contains:
# - scattering angle sampling
# - Henyey–Greenstein phase function
# - tabulated phase functions (CDF)
# - photon direction update
# - generation of the new mean free path

# This module is called after each scattering event
# when the polarized mode is disabled.

# ======================================================================================================================


import math
import random

from config import *
from io_materials import binaryTreeSearch






def scatter(photon):
    
    g = photon.g 
 
    if g is not None: # Henyey–Greenstein phase function
        costheta = g
        if abs(g) == 1:
            costheta = g
        elif abs(g) <= math.sqrt(eps):
            costheta = 2.0 * random.random() - 1.0
        else:
            costheta = (1 + g**2- ((1 - g**2) / (1 - g + 2 * g * random.random()))**2) / (2 * g)
            
 
    else:  # Custom phase function
        cdf = photon.phase_data["cdf"]
        CDFsize = photon.phase_data["size"]  
        jtheta = binaryTreeSearch ( random.random(), cdf)
        costheta = math.cos((jtheta + random.random())*math.pi/(CDFsize - 1));


        
    sintheta = math.sqrt(1 - costheta * costheta);
    phi = 2*math.pi*random.random();
    cosphi =  math.cos(phi);
    sinphi =  math.sin(phi);
    

    
    if abs(photon.uz) < 1:
        ux_temp =  sintheta*(photon.ux * photon.uz * cosphi - photon.uy * sinphi) / math.sqrt(photon.ux * photon.ux + photon.uy * photon.uy) + photon.ux * costheta
        uy_temp =  sintheta*(photon.uy * photon.uz * cosphi + photon.ux * sinphi) / math.sqrt(photon.ux * photon.ux + photon.uy * photon.uy) + photon.uy * costheta
        photon.uz = -1.0 * sintheta * cosphi * math.sqrt(photon.ux * photon.ux + photon.uy * photon.uy) + photon.uz * costheta
        photon.uy = uy_temp
        photon.ux = ux_temp
    else:
        if photon.uz > 0:
            signeuz = 1
        elif photon.uz < 0:
            signeuz = -1
        else:
            signeuz = 0
        photon.ux = sintheta * cosphi 
        photon.uy = sintheta * sinphi 
        photon.uz = costheta * signeuz
    
    # Update the photon direction vector in the class
    photon.update_voxel()
    
    # Recompute voxel distances   
    rn = random.random()
    photon.stepLeft = -math.log(rn)
