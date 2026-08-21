# MC_PYROUX_V1 : montecarlo core + one material + pencil beam

import math
import random
from dataclasses import dataclass

# ==========================================================
# Paramètres globaux 
# ==========================================================
Nphoton= 100000		# nombre de photons à simuler
mua = 1e-2		# absorption en [cm^-1]
mus = 0.1 		# diffusion en [cm^-1]
g_in = 0.9          # anisotrpie 0 < g <= 1
xSize = 100		# [cm] boite va de -xSize à xSize
ySize = 100  	# [cm] boite va de -ySize à ySize
zSize = 75		# [cm] boite va de 0 à zSize
Nx = 100		# nombre de voxel en x
Ny = 100		# nombre de voxel en y
Nz = 100 		# nombre de voxel en z
dx = 2 * xSize / Nx	# taille voxel en x
dy = 2 * ySize / Ny	# taille voxel en y 
dz = zSize / Nz		# taille voxel en z 
weightmin = 0.001 	# energie seuille pour roulette 
chance = 0.1 		# probabilité pourabsorber photon à la roulette
eps = 1e-10	# erreur spatial mininum propagation

# parametres de source
xinit = 0.0		# position en x de la source en [cm]
yinit = 0.0		# position en y de la source en [cm]
zinit = 0.0		# position en z de la source en [cm]
uxinit = 0		# direction de la source en x
uyinit = 0		# direction de la source en y
uzinit = 1		# direction de la source en z
# Si utilise source gaussienne 
W0 = 2.54		# Waist de la source en [cm]
theta = 2e-3		# Divergence de la source en [rad]




# parametre de Bilan

compteurR = 0
compteurT = 0
compteurA1 = 0

# ==========================================================
# Photon
# ==========================================================
@dataclass
class Photon:
	x: float
	y: float
	z: float

	ux: float
	uy: float
	uz: float

	g: float = g_in


	weight: float = 1.0
	alive: bool = True
	sameVoxel: bool = True
	stepLeft: float = 0.0
	
	
	
	# attributs calculés

	ix: int = 0
	iy: int = 0
	iz: int = 0

	tx: float = float("inf")
	ty: float = float("inf")
	tz: float = float("inf")

	def __post_init__(self):
		self.stepLeft = -math.log(random.random())
		self.update_voxel()

 
	def update_voxel(self):
		self.ix = int((self.x + xSize) / dx)
		self.iy = int((self.y + ySize) / dy)
		self.iz = int(self.z / dz)

		# -------------------------
		# voxel boundaries
		# -------------------------
		
		x_min = -xSize + self.ix * dx
		x_max = x_min + dx

		y_min = -ySize + self.iy * dy
		y_max = y_min + dy

		z_min = self.iz * dz
		z_max = z_min + dz
		
		if (self.ux > 0):
			self.tx = (x_max - self.x) / self.ux
		elif (self.ux < 0):
			self.tx = (x_min - self.x) / self.ux
		else:
			self.tx = float("inf") # eviter division par 0
			
		if (self.uy > 0):
			self.ty = (y_max - self.y) / self.uy
		elif (self.uy < 0):
			self.ty = (y_min - self.y) / self.uy
		else:
			self.ty = float("inf") # eviter division par 0
		
			
		if (self.uz > 0):
			self.tz = (z_max - self.z) / self.uz
		elif (self.uz < 0):
			self.tz = (z_min - self.z) / self.uz
		else:
			self.tz = float("inf") # eviter division par 0
	

# ==========================================================
# fonctions photon

def escape(photon):
	"""
	Bords absorbants :
	- si le photon sort par z < 0  => réflexion
	- si le photon sort par z > zSize => transmission
	- si sort en x ou y => perdu
	"""

	global compteurR, compteurT, compteurA1

	# réflexion (haut)
	if photon.z < 0:
		photon.alive = False
		compteurR += photon.weight
		return "R"

	# transmission (bas)
	elif photon.z > zSize:
		photon.alive = False
		compteurT += photon.weight
		return "T"

	# sortie latérale
	elif photon.x < -xSize or photon.x > xSize:
		photon.alive = False
		compteurA1 +=photon.weight
		return "X"

	elif photon.y < -ySize or photon.y > ySize:
		photon.alive = False
		compteurA1 +=photon.weight
		return "Y"

	# toujours dedans
	return "inside"



def roulette(photon):
	"""
	Roulette russe :
	si poids trop faible, soit survit, soit meurt
	"""
	global compteurA1
	if photon.weight < weightmin:

		rn = random.random()

		if rn <= chance:
			photon.weight /= chance   # survit
		else:
			photon.alive = False	  # meurt
			compteurA1 +=photon.weight

def propagate(photon):

	photon.sameVoxel = True;
	s = min(photon.stepLeft/mus, min(photon.tx, min(photon.ty ,photon.tz )) )
	
	

	if (s == photon.stepLeft/mus):
		photon.stepLeft = 0.0
	else :
		photon.stepLeft -= s * mus



	
	
	old_ix = photon.ix
	old_iy = photon.iy
	old_iz = photon.iz
	#propagation
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
		
	

	"""
	Absorption du photon sur une distance s
	"""
	absorb = -photon.weight * math.expm1(-mua * s)
	photon.weight -= absorb
	
	
	#recalcul position voxel et leur distance
	photon.update_voxel()
	
def scatter(photon):
    
    g = photon.g 
    if (g >= 0): # alors fonction de phase HG
        costheta = g
        if abs(g) == 1:
            costheta = g
        elif abs(g) <= math.sqrt(eps):
            costheta = 2.0 * random.random() - 1.0
        else:
            costheta = (1 + g**2- ((1 - g**2) / (1 - g + 2 * g * random.random()))**2) / (2 * g)

        
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
    
    # maj du nouveau vecteur direction du photon dans la classe
    photon.update_voxel()
    
    # recalcul voxel distances    
    rn = random.random()
    photon.stepLeft = -math.log(rn)
    

# ==========================================================
# Coeur Montecarlo 

for inphoton in range (1,Nphoton+1):
	print(f"\rProgress: {inphoton / Nphoton * 100:.1f}%", end="", flush=True)
	p = Photon(xinit,yinit,zinit,uxinit,uyinit,uzinit) # pencil beam


	
	while (p.alive ):
		while (p.alive == True and p.stepLeft > 0):
			propagate(p) # propagation + absorption	
			if (p.sameVoxel == False):
				escape(p) # verifier frontiere
		roulette(p) # eleminer photon de faible energy ou leur laisser une chance de survie 
		scatter(p) # scattering du photon changement de ux uy uz

	

# ==========================================================
# Bilan

print ("\n Nombre de photons émis dans la simulation : ", Nphoton)

print( "\n Bilan par nombre de photons:")
print( "Reflectance R = ", compteurR)
print( "Transmittance T = ", compteurT)
#print( "Absorbance A1 = ", compteurA1) # on compte à la main les photons absorbés
print( "Absorbance A = ", (Nphoton-(compteurR+compteurT))) # on fait la différence pour connaitre photon	



print( "\n Bilan par taux:")	
print( "Reflectance R = ", compteurR/Nphoton)
print( "Transmittance T = ", compteurT/Nphoton)
#print( "Absorbance A1 = ", compteurA1/Nphoton) # on compte à la main les photons absorbés
print( "Absorbance A = ", (1-(compteurR+compteurT)/Nphoton)) # on fait la différence pour connaitre photon absorbés

# ==========================================================




