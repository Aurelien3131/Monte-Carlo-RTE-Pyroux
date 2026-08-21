# MC_PYROUX_V2 : montecarlo core + one material + pencil beam + gaussian beam + detector

import math
import random
from dataclasses import dataclass

# ==========================================================
# Paramètres globaux 
# ==========================================================
Nphoton = 100000	# nombre de photons à simuler
mua = 1e-5		# absorption en [cm^-1]
mus = 1e-4		# diffusion en [cm^-1]
g_in = 0.9          # anisotrpie 0 < g <= 1
xSize = 100		# [cm] boite va de -xSize à xSize
ySize = 100		# [cm] boite va de -ySize à ySize
zSize = 50		# [cm] boite va de 0 à zSize
Nx = 100		# nombre de voxel en x
Ny = 100		# nombre de voxel en y
Nz = 100 		# nombre de voxel en z
dx = 2 * xSize / Nx	# taille voxel en x
dy = 2 * ySize / Ny	# taille voxel en y 
dz = zSize / Nz		# taille voxel en z 
weightmin = 0.001 	# energie seuille pour roulette 
chance = 0.1 		# probabilité pour absorber photon à la roulette
eps = 1e-10		# erreur spatial mininum propagation

# parametres de source
xinit = 0.0		# position en x de la source en [cm]
yinit = 0.0		# position en y de la source en [cm]
zinit = 0.0		# position en z de la source en [cm]
zlaunch = zinit
uxinit = 0		# direction de la source en x
uyinit = 0		# direction de la source en y
uzinit = 1		# direction de la source en z
# Si utilise source gaussienne 
W0 = 2.54		# Waist de la source en [cm]
theta = 2e-3		# Divergence de la source en [rad]

# initialisation détecteur 

xdet = 0.0 		# position du détecteur en x en [cm]
ydet = 0.0 		# position du détecteur en y en [cm]
zdet = zSize		# position du détecteur en z en [cm]
thetadet = math.pi	#orientation du detecteur -> dans le plan xy et regarde vers -z
phidet = 0		# orientation du detecteur
diametre = 15 		# diametre du detecteru en [cm]
NA = 0.5 		# ouverture numérique du détecteur theta = asin(NA)




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

	def __post_init__(self):
		self.nz = math.cos(self.theta)
		self.nx = math.sin(self.theta) * math.cos(self.phi)
		self.ny = math.sin(self.theta) * math.sin(self.phi)

		self.theta_max = math.asin(self.NA)





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

	# réflexion 
	if photon.z < 0:
		photon.alive = False
		compteurR += photon.weight
		return "R"

	# transmission
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
# fonctions laser

def axisrotate(r, u, theta):
	"""
	Rotation du vecteur r autour de l’axe unitaire u d’un angle theta
	(copie de MCmatlab)
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

    # choisir un vecteur non parallèle
    if abs(uz) < 0.9:
        ref = (0.0, 0.0, 1.0)
    else:
        ref = (1.0, 0.0, 0.0)

    rx, ry, rz = ref

    # produit vectoriel ref × u
    vx = ry*uz - rz*uy
    vy = rz*ux - rx*uz
    vz = rx*uy - ry*ux

    norm = math.sqrt(vx*vx + vy*vy + vz*vz)

    return (vx/norm, vy/norm, vz/norm)

def gaussian_source(x0, y0, z0, w0, divergence,z_launch, ux,uy,uz):

	u = (0 ,0 ,1 ) # (ux , uy, uz) # vecteur direction laser
	v = (1 ,0 ,0 ) #orthogonal_unit_vector(u) # vecteur ortogonale au laser	
	
	# 1. Cible dans le waist : NEAR FIELD
	w0_vec = axisrotate(v, u, 2.0 * math.pi * random.random())
	r = w0 * math.sqrt(-0.5 * math.log(random.random()))
	
	xt = x0 + r * w0_vec[0]
	yt = y0 + r * w0_vec[1]
	zt = z0 + r * w0_vec[2]
	

	# 2. Direction (divergence) : FAR FIELD
	w0_vec = axisrotate(v, u, 2.0 * math.pi * random.random()) 
	phi = math.atan(math.tan(divergence) * math.sqrt(-0.5 * math.log(random.random())))
	
	ux, uy, uz =  axisrotate(u, w0_vec, phi)
	
	# 3. Projection dans le plan z = z_launch
	if abs(uz) < 1e-12: # sécurité numérique
		uz = 1e-12
	x = xt - (zt - z_launch) * ux / uz
	y = yt - (zt - z_launch) * uy / uz
	z = 0#z_launch
	return Photon(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz)
	

# ==========================================================
# fonction detecteur

def detect(photon, det):
    if (photon.z >= det.z): # le photon depasse le detecteur ou est sur le detecteur
        
        if abs(photon.uz) < 1e-12:
            return False    
    
        # 1. Remonter le photon dans le plan du détecteur
        t = (det.z - photon.z) / photon.uz # pas de temps négatif -> t >= 0 ssi uz > 0
        xi = photon.x + t * photon.ux
        yi = photon.y + t * photon.uy    
    
        # 2. Condition position : le photon est-il entré dans le diametre de la fibre ?
        dist = math.sqrt((xi - det.x)**2 + (yi - det.y)**2) # distance entre le centre de la fibre et la position du photon dans le plan
        if dist > det.diam / 2:
            return False

        # 3. Condition angulaire : le photon est-il dans le cone d'ouverture angulaire ?
        theta_max = math.asin(det.NA)  # en radians # angle d'acceptation max d'un photon
        norm_uphoton = math.sqrt(photon.ux**2 + photon.uy**2 + photon.uz**2) # norme du photon direction
        scalaire_photon_to_normale = 1.0 * photon.uz # produit scalaire entre le photon et la normale au detecteur
        thetaphoton =  math.acos(scalaire_photon_to_normale/norm_uphoton)
        
        
        if thetaphoton > theta_max:
            return False
        
        # 4. Alors on collecte le photon dans le détecteur
        det.collected += photon.weight
        return True



# ==========================================================

# Coeur Montecarlo 

detector = Detector(xdet,ydet,zdet,thetadet, phidet ,diametre,  NA ) #initialisation du detecteur

#f = open("V2_photons_init.csv", "w")


for inphoton in range (1,Nphoton+1):
	print(f"\rProgress: {inphoton / Nphoton * 100:.1f}%", end="", flush=True)
	#p = Photon(xinit,yinit,zinit,uxinit,uyinit,uzinit) # pencil beam
	p = gaussian_source(xinit,yinit,zinit, W0, theta, zinit, uxinit, uyinit, uzinit) # source gaussienne
#	f.write(f"{p.x},{p.y},{p.z},{p.ux},{p.uy},{p.uz}\n")
	
	while (p.alive ):
		while (p.alive == True and p.stepLeft > 0):
			propagate(p) # propagation + absorption	
			if (p.sameVoxel == False):
				detect(p,detector)
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

print ( "\nResulat collecteur:")
print( "Nombre de photons dans le collecteur", detector.collected )
print( "Taux de photons dans le collecteur", detector.collected / Nphoton)
print( "% de photons dans le collecteur", (detector.collected / Nphoton) * 100)

# ==========================================================



