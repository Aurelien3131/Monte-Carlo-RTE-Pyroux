# ======================================================================================================================

# geometry.py

# Monte Carlo simulation geometry

# Contains:
# - detector definition
# - photon collection
# - time histograms
# - voxelized box definition
# - assignment of local optical properties


# Initial geometric convention:

# Source (z=0)                     Detector (z=zSize)
# ●──────────────────────────────► +Z


# Domain:
# x ∈ [-xSize ; +xSize]
# y ∈ [-ySize ; +ySize]
# z ∈ [0 ; zSize]

# ======================================================================================================================


import math
from dataclasses import dataclass, field

import numpy as np

from config import *
from io_materials import (load_phase_function, load_polar_data)


@dataclass
class Detector:
    x: float
    y: float
    z: float
    theta: float
    phi: float
    diam: float
    NA: float

    collected: float = 0.0

    D_I: float = 0.0
    D_Q: float = 0.0
    D_U: float = 0.0
    D_V: float = 0.0

    def __post_init__(self):
        self.nz = math.cos(self.theta)
        self.nx = math.sin(self.theta) * math.cos(self.phi)
        self.ny = math.sin(self.theta) * math.sin(self.phi)

        self.theta_max = math.asin(self.NA)
        

        	
def detect(photon, det, histo,): # Function designed for a sensor that is always perpendicular to the z-axis      
    if abs(photon.uz) < 1e-12: # Photons parallel to the detector are not taken into account
         return False  
    
    # TRANSMISSION
    if (det.nz == -1) : # If the detector is facing z = 0
    	if (photon.z >= det.z): # The photon either passes by the detector or hits the detector
        	# 1. Bring the photon back into the detector plane
        	t = ((det.z - photon.z) / photon.uz)* photon.n /(100*C_VELOCITY) # Negatif time step -> t >= 0 only if uz > 0
        	xi = photon.x + t * photon.ux
        	yi = photon.y + t * photon.uy    
        
        	# 2. Position condition: Did the photon enter the fiber's diameter?
        	dist = math.sqrt((xi - det.x)**2 + (yi - det.y)**2) 
        	# That is the distance between the center of the fiber and the position of the photon in the plane
        	if dist > det.diam / 2:
        	    return False

        	# 3. Angular condition: Is the photon within the angular aperture?
        	theta_max = math.asin(det.NA)  # rad
        	# Maximum acceptance angle of a photon
        	
        	norm_uphoton = math.sqrt(photon.ux**2 + photon.uy**2 + photon.uz**2) # Photon direction norm
        	scalaire_photon_to_normale = -1.0*det.nz * photon.uz # Scalar product between the photon and the normal to the detector
        	thetaphoton =  math.acos(scalaire_photon_to_normale/norm_uphoton)
        	thetaphoton =  math.acos(scalaire_photon_to_normale/norm_uphoton)
        
        
        	if thetaphoton > theta_max:
       	     		return False
        
       		# 4. Then we collect the photon in the detector
        	det.collected += photon.weight
        	det.D_I += photon.I * photon.weight
        	det.D_Q += photon.Q * photon.weight
        	det.D_U += photon.U * photon.weight
        	det.D_V += photon.V * photon.weight
        	# 5. We update the time the photon has elapsed since passing the sensor and add it to the time histogram
        	arrival_time = photon.timephot + t 
        	
        	histo.add_photon(photon, arrival_time)
        
        	return True
    # REFLEXION
    if (det.nz == 1) : 
    	if (photon.z <= det.z): 
        	# 1. 
        	t = ((det.z - photon.z) / photon.uz)* photon.n /(100*C_VELOCITY)
        	xi = photon.x + t * photon.ux
        	yi = photon.y + t * photon.uy    
        
        	# 2. 
        	dist = math.sqrt((xi - det.x)**2 + (yi - det.y)**2)
        	if dist > det.diam / 2:
        	    return False

        	# 3.
        	theta_max = math.asin(det.NA)  # rad
        	norm_uphoton = math.sqrt(photon.ux**2 + photon.uy**2 + photon.uz**2)
        	scalaire_photon_to_normale = -1.0*det.nz * photon.uz 
        	thetaphoton =  math.acos(scalaire_photon_to_normale/norm_uphoton)
        
        
        	if thetaphoton > theta_max:
       	     		return False
        
       		# 4. 
        	det.collected += photon.weight
        	det.D_I += photon.I * photon.weight
        	det.D_Q += photon.Q * photon.weight
        	det.D_U += photon.U * photon.weight
        	det.D_V += photon.V * photon.weight
        	# 5. 
        	arrival_time = photon.timephot + t 
        	
        	histo.add_photon(photon, arrival_time)
        
        	return True            

@dataclass
class Histogramme:
    time_in: float = 0.0
    time_out: float = 1.0
    time_bin: int = 100

    bins: np.ndarray = field(init=False)
    hist: np.ndarray = field(init=False)
    numberphot: np.ndarray = field(init=False)
    
    underflow_t: int = field(init=False)   # time
    overflow_t: int = field(init=False)
    
    underflow: float = field(init=False)   # Before the time-window (weight)
    overflow: float = field(init=False)    # After the time-window (weight)

    underflow_n: int = field(init=False)   # Photons Number
    overflow_n: int = field(init=False)
    
    def __post_init__(self):
        self.bins = np.linspace(self.time_in, self.time_out, self.time_bin + 1)
        self.hist = np.zeros(self.time_bin)
        self.numberphot = np.zeros(self.time_bin)
        
        self.underflow_t = 0
        self.overflow_t = 0
        self.underflow = 0.0
        self.overflow = 0.0
        self.underflow_n = 0
        self.overflow_n = 0

    def add_photon(self, photon, arrival_time = None):
        if arrival_time == None :
            t = photon.timephot
        else : 
            t = arrival_time
        # BEFORE TIME WINDOW
        if t < self.time_in:
            self.underflow_t = t
            self.underflow += photon.weight
            self.underflow_n += 1.0
            return

        # AFTER TIME WINDOW
        if t >= self.time_out:
            self.overflow_t = t
            self.overflow += photon.weight
            self.overflow_n += 1.0
            return

        # Find the bin
        idx = int((t - self.time_in) / (self.time_out - self.time_in) * self.time_bin)

        if 0 <= idx < self.time_bin:
            self.hist[idx] += photon.weight
            self.numberphot[idx] += 1


    def get(self):
        centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        return centers, self.hist, self.numberphot



@dataclass
class HistogrammeEmission:
    time_in: float
    time_out: float
    time_bin: int

    def __post_init__(self):
        self.bins = np.linspace(self.time_in,
                                self.time_out,
                                self.time_bin + 1)
        self.hist = np.zeros(self.time_bin)

    def add_emit(self, photon):

        t = photon.t_emit

        if t < self.time_in or t >= self.time_out:
            return

        idx = int(
            (t - self.time_in)
            / (self.time_out - self.time_in)
            * self.time_bin
        )

        if 0 <= idx < self.time_bin:
            self.hist[idx] += photon.weight

    def get(self):
        centers = 0.5*(self.bins[:-1] + self.bins[1:])
        return centers, self.hist
        
        

@dataclass
class Box:
    xSize: float  # [cm] box ranges from -xSize to xSize
    ySize: float  # [cm] box ranges from -ySize to ySize
    zSize: float  # [cm] box ranges from 0 to zSize
    Nx: int
    Ny: int
    Nz: int

    def __post_init__(self):
        self.dx = 2 * self.xSize / self.Nx
        self.dy = 2 * self.ySize / self.Ny
        self.dz = self.zSize / self.Nz

        # List of materials: default index 0
        self.materials = [{"mua": 0.0, "mus": 0.0, "g": 0.0, "n": 1.0 ,"phase_file":None, "phase_data":None,"polar_file":None, "polar_data":None}]

        # voxel_map :  everything initialized to material 0
        self.voxel_map = [[[0 for _ in range(self.Nz)]
                              for _ in range(self.Ny)]
                              for _ in range(self.Nx)]

    def add_material(self, mua, mus, g, n, phase_file=None, polar=False):

        material = {"mua": mua, "mus": mus, "g": g, "n": n, "phase_data": None, "polar": polar, "polar_data": None}
        if phase_file is not None:
            material["phase_data"] = load_phase_function(phase_file,2000)
        if polar:
            material["polar_data"] = load_polar_data(phase_file,2000)

        self.materials.append(material)

        return len(self.materials) - 1

    def set_voxel(self, ix, iy, iz, material_idx):
        self.voxel_map[ix][iy][iz] = material_idx

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
        if (photon.ix < 0 or photon.ix >= self.Nx or photon.iy < 0 or photon.iy >= self.Ny or  photon.iz < 0 or photon.iz >= self.Nz):
            photon.alive = False
            return 
        else:
            idx = self.voxel_map[photon.ix][photon.iy][photon.iz]

            photon.mua = self.materials[idx]["mua"]
            photon.mus = self.materials[idx]["mus"]
            photon.g   = self.materials[idx]["g"]
            photon.n   = self.materials[idx]["n"]
            photon.phase_data = self.materials[idx]["phase_data"]
            photon.polar = self.materials[idx]["polar"]
            photon.polar_data = self.materials[idx]["polar_data"]
