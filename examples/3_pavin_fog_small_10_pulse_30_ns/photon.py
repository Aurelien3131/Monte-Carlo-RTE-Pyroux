# ======================================================================================================================

# photon.py

# Definition of the Photon class and its state during the simulation.

# Contains:
# - photon position and direction
# - propagation and emission time
# - local optical properties
# - polarization parameters (Stokes)
# - Monte Carlo statistical weight
# - voxel coordinates and interface distances


# This module centralizes all information carried
# and updated during photon propagation.



# ======================================================================================================================



import math
import random

from dataclasses import dataclass

from config import *

@dataclass
class Photon:
	# Position
	x: float
	y: float
	z: float
	
	dz: float
    
	# Direction
	ux: float
	uy: float
	uz: float
	
	# Photon time
	timephot: float = 0.0
	freq: float = 1.0 # Modulation frequency
	t_emit: float = 0.0
	
	# Optical parameters
	mus: float = 0.0
	mua: float = 0.0
	n: float = 0.0 
	g: float = 0.0

	polar: bool = False
	polar_data: object = None

	I: float = 1
	Q: float = 0
	U: float = 0
	V: float = 0

	ex1: float = 1.0
	ey1: float = 0.0
	ez1: float = 0.0

	ex2: float = 0.0
	ey2: float = 1.0
	ez2: float = 0.0
	#e1 ⟂ u
	#e2 = u × e1

	weight: float = 1.0
	alive: bool = True
	sameVoxel: bool = True
	stepLeft: float = 0.0
	
	phase_data: object = None
	
	# Computed attributes

	ix: int = 0
	iy: int = 0
	iz: int = 0

	tx: float = float("inf")
	ty: float = float("inf")
	tz: float = float("inf")

	def __post_init__(self):
		self.stepLeft = -math.log(random.random())
		self.timephot = self.t_emit
		self.update_voxel()
		self.init_polar_basis()


 
 
 
	def update_voxel(self):
		self.ix = int((self.x + xSize) / dx)
		self.iy = int((self.y + ySize) / dy)
		self.iz = int(self.z / self.dz)

		# -------------------------
		# voxel boundaries
		# -------------------------
		
		x_min = -xSize + self.ix * dx
		x_max = x_min + dx

		y_min = -ySize + self.iy * dy
		y_max = y_min + dy

		z_min = self.iz * self.dz
		z_max = z_min + self.dz
		
		if (self.ux > 0):
			self.tx = (x_max - self.x) / self.ux
		elif (self.ux < 0):
			self.tx = (x_min - self.x) / self.ux
		else:
			self.tx = float("inf") # Avoid division by zero
			
		if (self.uy > 0):
			self.ty = (y_max - self.y) / self.uy
		elif (self.uy < 0):
			self.ty = (y_min - self.y) / self.uy
		else:
			self.ty = float("inf") # Avoid division by zero
		
			
		if (self.uz > 0):
			self.tz = (z_max - self.z) / self.uz
		elif (self.uz < 0):
			self.tz = (z_min - self.z) / self.uz
		else:
			self.tz = float("inf") # Avoid division by zero
	
	def init_polar_basis(self):

		# Photon direction
		uz = self.uz

		# Choose a non-parallel axis
		if abs(uz) < 0.9:
			ax, ay, az = 0,0,1
		else:
			ax, ay, az = 1,0,0

		# e1 = axe × u
		ex = ay*self.uz - az*self.uy
		ey = az*self.ux - ax*self.uz
		ez = ax*self.uy - ay*self.ux

		norm = math.sqrt(ex*ex+ey*ey+ez*ez)

		self.ex1 = ex/norm
		self.ey1 = ey/norm
		self.ez1 = ez/norm

		# e2 = u × e1
		self.ex2 = self.uy*self.ez1 - self.uz*self.ey1
		self.ey2 = self.uz*self.ex1 - self.ux*self.ez1
		self.ez2 = self.ux*self.ey1 - self.uy*self.ex1
    
