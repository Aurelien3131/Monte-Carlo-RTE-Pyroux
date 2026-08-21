# MC_PYROUX_V5 : montecarlo core + one material + pencil beam + gaussian beam + detector + multimaterial + plane wave + condition cyclique x/y
# + custom phase function + polar
# attention dans cette version les conditions periodiques n'ont pas de sécurité en cas de propagation perpendiculaire à l'axe z donc un photon se propageant parfaitement dans le plan xy peut bloquer le code
import math
import random
from dataclasses import dataclass
import bisect
import numpy as np
from scipy.interpolate import interp1d

# ==========================================================
# Paramètres globaux 
# ==========================================================
Nphoton = 10000	# nombre de photons à simuler

xSize = 100		# [cm] boite va de -xSize à xSize
ySize = 100		# [cm] boite va de -ySize à ySize
zSize = 2500	# [cm] boite va de 0 à zSize
Nx = 100		# nombre de voxel en x
Ny = 100		# nombre de voxel en y
Nz = 100 		# nombre de voxel en z
dx = 2 * xSize / Nx	# taille voxel en x
dy = 2 * ySize / Ny	# taille voxel en y 
dz = zSize / Nz		# taille voxel en z 
weightmin = 0.001 	# energie seuille pour roulette 
chance = 0.1 		# probabilité pourabsorber photon à la roulette
eps = 1e-10		# erreur spatial minium propagation

# parametres entrées de polarisation 

I_in = 1    # doit etre 1
Q_in = 0
U_in = 1
V_in = 0

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
NA = 0.1 		# ouverture numérique du détecteur theta = asin(NA)




# parametre de Bilan

compteurR = 0
compteurT = 0
compteurA1 = 0

R_I = R_Q = R_U = R_V = 0.0     # reflexion face entrée
T_I = T_Q = T_U = T_V = 0.0     # transmission face sortie

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
    # vecteur direction u
    
	mus: float = 0.0
	mua: float = 0.0
	n: float = 0.0 
	g: float = 0.0

	polar: bool = False
	polar_data: object = None

	I: float = I_in
	Q: float = Q_in
	U: float = U_in
	V: float = V_in

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
		self.init_polar_basis()

 
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
	
	def init_polar_basis(self):

		# direction photon
		uz = self.uz

		# choisir axe non parallèle
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
    #transmission polar
	D_I: float = 0.0
	D_Q: float = 0.0
	D_U: float = 0.0
	D_V: float = 0.0

	def __post_init__(self):
		self.nz = math.cos(self.theta)
		self.nx = math.sin(self.theta) * math.cos(self.phi)
		self.ny = math.sin(self.theta) * math.sin(self.phi)

		self.theta_max = math.asin(self.NA)

# ==========================================================

@dataclass
class Box:
    xSize: float  # [cm] boite va de -xSize à xSize
    ySize: float  # [cm] boite va de -ySize à ySize
    zSize: float  # [cm] boite va de 0 à zSize
    Nx: int
    Ny: int
    Nz: int

    def __post_init__(self):
        self.dx = 2 * self.xSize / self.Nx
        self.dy = 2 * self.ySize / self.Ny
        self.dz = self.zSize / self.Nz

        # liste des matériaux : index 0 par défaut
        self.materials = [{"mua": 0.0, "mus": 0.0, "g": 0.0, "n": 1.0 ,"phase_file":None, "phase_data":None,"polar_file":None, "polar_data":None}]

        # voxel_map : tout initialisé au matériau 0
        self.voxel_map = [[[0 for _ in range(self.Nz)]
                              for _ in range(self.Ny)]
                              for _ in range(self.Nx)]

    def add_material(self, mua, mus, g, n, phase_file=None, polar=False):

        material = {
        "mua": mua,
        "mus": mus,
        "g": g,
        "n": n,
        "phase_file": phase_file,
        "phase_data": None,
        "polar": polar,
        "polar_data": None
        }

        if phase_file is not None:
            material["phase_data"] = load_phase_function(phase_file,2000)
        if polar:
            material["polar_data"] = load_polar_data(phase_file,2000)

        self.materials.append(material)

        return len(self.materials) - 1

    def set_voxel(self, ix, iy, iz, material_idx):
        """Assigne un matériau à un voxel"""
        self.voxel_map[ix][iy][iz] = material_idx

    def set_region(self, x_range, y_range, z_range, material_idx):
        """Assigne un matériau à une région en cm
        x_range = (xmin, xmax), y_range = (ymin, ymax), z_range = (zmin, zmax)"""
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
            return # photon hors grille  il est mort x.x
        else:
            idx = self.voxel_map[photon.ix][photon.iy][photon.iz]	#trouve la position du voxel

            photon.mua = self.materials[idx]["mua"]		# donne les propriétés optiques du voxel au photon 
            photon.mus = self.materials[idx]["mus"]
            photon.g   = self.materials[idx]["g"]
            photon.n   = self.materials[idx]["n"]
            photon.phase_data = self.materials[idx]["phase_data"]   # fonction de phase du materiau
            photon.polar = self.materials[idx]["polar"]
            photon.polar_data = self.materials[idx]["polar_data"]
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
	global R_I, R_Q, R_U, R_V
	global T_I, T_Q, T_U, T_V
	# réflexion 
	if photon.z < 0:
		photon.alive = False
		compteurR += photon.weight
		R_I += photon.I * photon.weight
		R_Q += photon.Q * photon.weight
		R_U += photon.U * photon.weight
		R_V += photon.V * photon.weight
		return "R"

	# transmission
	elif photon.z > zSize:
		photon.alive = False
		compteurT += photon.weight
		T_I += photon.I * photon.weight
		T_Q += photon.Q * photon.weight
		T_U += photon.U * photon.weight
		T_V += photon.V * photon.weight
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

def escape_periodic_xy(photon):
	"""
	Bords périodiques en x et y :
	- z < 0        => réflexion
	- z > zSize    => transmission
	- x, y         => périodiques (wrap)
	"""

	global compteurR, compteurT
	global R_I, R_Q, R_U, R_V
	global T_I, T_Q, T_U, T_V

	# réflexion
	if photon.z < 0:
		photon.alive = False
		compteurR += photon.weight
		R_I += photon.I * photon.weight
		R_Q += photon.Q * photon.weight
		R_U += photon.U * photon.weight
		R_V += photon.V * photon.weight
		return "R"

	# transmission
	elif photon.z > zSize:
		photon.alive = False
		compteurT += photon.weight
		T_I += photon.I * photon.weight
		T_Q += photon.Q * photon.weight
		T_U += photon.U * photon.weight
		T_V += photon.V * photon.weight
		return "T"

	# conditions périodiques en x
	if photon.x > xSize:
		photon.x -= 2 * xSize
	elif photon.x < -xSize:
		photon.x += 2 * xSize

	# conditions périodiques en y
	if photon.y > ySize:
		photon.y -= 2 * ySize
	elif photon.y < -ySize:
		photon.y += 2 * ySize

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
    
	s = min(photon.stepLeft/photon.mus, min(photon.tx, min(photon.ty ,photon.tz )) )
        
        
	if (s == photon.stepLeft/photon.mus):
		photon.stepLeft = 0.0
	else :
		photon.stepLeft -= s * photon.mus


	# 1. Propagation
	old_ix = photon.ix
	old_iy = photon.iy
	old_iz = photon.iz

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

	# 2. Snell-Descartes / Fresnel
	if photon.sameVoxel == False :
	 
		
		# nouveau voxel 
		ix = max(0, min(Nx - 1, photon.ix))
		iy = max(0, min(Ny - 1, photon.iy))
		iz = max(0, min(Nz - 1, photon.iz))
		# propriétés du nouveau voxel
		new_idx = box.voxel_map[ix][iy][iz]
		new_n = box.materials[new_idx]["n"]
		mu = old_n / new_n
		
		# 2 cas de mu : 
		# mu!=1:
		if mu != 1: # indices optiques sont différents donc il faut appliquer loi de refraction/reflexion
			photonReflected = False

		
			nx, ny, nz = 0.0, 0.0, 0.0

			if abs(s - photon.tx) < eps:
				nx = -1 if photon.ux > 0 else 1
			elif abs(s - photon.ty) < eps:
				ny = -1 if photon.uy > 0 else 1
			elif abs(s - photon.tz) < eps:
				nz = -1 if photon.uz > 0 else 1

			# dot product
			cos_in = nx*photon.ux + ny*photon.uy + nz*photon.uz

			if cos_in > 0:
				
				cos_out_sqr = 1 - mu**2 * (1 - cos_in**2)

				if cos_out_sqr > 0:

					cos_out = math.sqrt(cos_out_sqr)
					R = (((mu*cos_in - cos_out)/(mu*cos_in + cos_out))**2 + ((mu*cos_out - cos_in)/(mu*cos_out + cos_in))**2) / 2
					photonReflected = random.random() <= R

			else:
				photonReflected = True  # réflexion totale

			if photonReflected:
				# réflexion
				photon.ux -= 2*nx*cos_in
				photon.uy -= 2*ny*cos_in
				photon.uz -= 2*nz*cos_in

			else:
				# réfraction
				cos_out = math.sqrt(max(0.0, cos_out_sqr))
				ncoeff = cos_out - mu*cos_in

				photon.ux = ncoeff*nx + mu*photon.ux
				photon.uy = ncoeff*ny + mu*photon.uy
				photon.uz = ncoeff*nz + mu*photon.uz
				photon.n = new_n  # mise à jour RI
		
		# mu == 1:
		if mu == 1: # meme indice de refraction on ne fait rien
			pass 
	# 3. Absorption
	absorb = -photon.weight * math.expm1(-photon.mua * s)
	photon.weight -= absorb
	
    
	#recalcul position voxel et leur distance
	photon.update_voxel()
	

def scatter(photon):
    
    g = photon.g 
 
    if g is not None: # alors fonction de phase HG
        costheta = g
        if abs(g) == 1:
            costheta = g
        elif abs(g) <= math.sqrt(eps):
            costheta = 2.0 * random.random() - 1.0
        else:
            costheta = (1 + g**2- ((1 - g**2) / (1 - g + 2 * g * random.random()))**2) / (2 * g)
            
 
    else: # fonction de phase custom
        cdf = photon.phase_data["cdf"]
        CDFsize = photon.phase_data["size"]  
        jtheta = binaryTreeSearch ( random.random(), cdf)
        costheta = math.cos((jtheta + random.random())*math.pi/(CDFsize - 1));
        #theta = photon.phase_data["thetas"][jtheta]
        #costheta = math.cos(theta)

        
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
# fonctions polarisation

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



    
def scatter_polarized(photon):

    pd = photon.polar_data
    s11_0 = float(pd["s11"](0.0))
    I_ref  = s11_0 * photon.I  # maximum de référence à θ=0

    while True:
        # tirage uniforme en cos(theta) — comme Ramella acos(2·rnd−1)
        costheta = 2.0 * random.random() - 1.0
        theta    = math.acos(costheta)
        phi      = 2.0 * math.pi * random.random()

        s11 = float(pd["s11"](theta))
        s12 = float(pd["s12"](theta))

        cos2phi = math.cos(2.0 * phi)
        sin2phi = math.sin(2.0 * phi)

        # critère complet incluant la polarisation courante
        I_test = s11 * photon.I + s12 * (photon.Q * cos2phi + photon.U * sin2phi)

        if random.random() * I_ref <= I_test:
            break

    sintheta = math.sqrt(max(0.0, 1.0 - costheta * costheta))
    cosphi   = math.cos(phi)
    sinphi   = math.sin(phi)

    # ROTATION 1 : Stokes → plan de diffusion
    rotate_photon_stokes(photon, phi)   # inchangé

    # MUELLER
    s33 = float(pd["s33"](theta))
    s43 = float(pd["s43"](theta))

    photon.I, photon.Q, photon.U, photon.V = mueller_scatter(
        photon.I, photon.Q, photon.U, photon.V, s11, s12, s33, s43
    )

    # MAJ DIRECTION (uz_old nécessaire pour rotation 2)
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

    # ── Nouveau step libre ───────────────────────────────────────────
    photon.update_voxel()
    photon.stepLeft = -math.log(random.random())

    
    
def update_polar_basis(photon): # met a jour le vecteur local de rotation du photon

    ux, uy, uz = photon.ux, photon.uy, photon.uz

    # choisir vecteur de référence
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


	u = (0 ,0 ,1)#(ux , uy, uz) # vecteur direction laser
	v = (1 ,0 ,0)#orthogonal_unit_vector(u) # vecteur ortogonale au laser	
	
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
	if abs(uz) < eps: # sécurité numérique
		uz = eps
	x = xt - (zt - z_launch) * ux / uz
	y = yt - (zt - z_launch) * uy / uz
	z = 0 # plan z_launch
	return Photon(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz)
	
def launch_plane_wave(zinit):
    # position uniforme sur la face z=0
    x = (random.random() - 0.5) * 2 * xSize
    y = (random.random() - 0.5) * 2 * ySize
    z = zinit

    # direction fixe (onde plane)
    ux = uxinit
    uy = uyinit
    uz = uzinit

    p = Photon(x, y, z, ux, uy, uz)

    # (optionnel) normalisation direction
    norm = math.sqrt(ux**2 + uy**2 + uz**2)
    p.ux /= norm
    p.uy /= norm
    p.uz /= norm
    return p
    
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
        det.D_I += photon.I * photon.weight
        det.D_Q += photon.Q * photon.weight
        det.D_U += photon.U * photon.weight
        det.D_V += photon.V * photon.weight
        return True
    
# ==========================================================
# Fonction chargement fonction de phase et contruction cdf

def load_phase_function(filename, max_elements=2000):
    """
    fichier :
        cos(theta)   phase(theta)

    retourne :
        {
            "cdf": [...],     # taille CDFSIZE+1
            "size": CDFSIZE
        }
    """

    mu_file = []
    phase_file = []

    # 1. Lecture fichier
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
        
    # 2. construction phase_func continue pou la cdf
    theta_deg = []
    for ij in range(0,len(mu_file)):
        theta_deg.append( math.acos(mu_file[ij]) * (180/math.pi) )
    
    theta_deg = np.flip(theta_deg)
    phase_file = np.flip(phase_file)
    phase_func = interp1d(theta_deg, phase_file, kind='linear')



    # 3. construction cdf 
    CDFsize = 200 #nombre d'element de la cdf # defini par mcmatlab
    thetas =[]
    for ik in range (1,CDFsize+1):
        thetas.append((math.pi/CDFsize)*(ik - 0.5))
    pdf = []
    for il in range(0, CDFsize):
        pdf.append(math.sin(thetas[il]) * phase_func(thetas[il] * 180/math.pi))
    S=sum(pdf)
    if S <= 0:
        raise ValueError("PDF invalid (sum <= 0)")
    
    pdf = [p / S for p in pdf]
    
    CDF = [0.0]
    cumul = 0.0
    for p in pdf:
        cumul += p
        CDF.append(cumul)

    CDF[-1] = 1.0  # sécurité



    return {"cdf": CDF, "size": CDFsize, "thetas": thetas}

def binaryTreeSearch(rand, cdf):
    # bisect_right trouve i tel que cdf[i-1] <= rand < cdf[i]
    # on veut j tel que cdf[j] < rand <= cdf[j+1], donc j = i-1
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

    if len(mu_vals) < 4:
        raise ValueError("Polar file too short")

    # theta croissant
    theta_vals = [math.acos(mu) for mu in mu_vals]

    # si mu va de +1 vers -1 alors theta déjà croissant
    # sinon on inverse
    if theta_vals[1] < theta_vals[0]:

        theta_vals.reverse()
        s11_vals.reverse()
        s12_vals.reverse()
        s33_vals.reverse()
        s43_vals.reverse()

    #pour la normalisation

    
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


# ==========================================================

# initialisation des classes 

box = Box(xSize, ySize, zSize, Nx, Ny, Nz) # initialisation taille de boite et voxel

# attention mua et mus doivent etre en cm^-1
# matériau 1 : materiau 1
#idx_mat1 = box.add_material(mua=1.50201609e-04, mus=2.59152578e-04, g=None, n=1, phase_file="ram_function_phases_Pav_al_small_10_v_40_l_9.8_nm.txt", polar=True) # g/phase_file = None desactiver
# matériau 2 : materiau 2
idx_mat2 = box.add_material(mua=1.26466852e-05, mus=2.19386800e-04, g=None, n=1, phase_file="ram_function_phases_PFog14_l_4_nm.txt", polar=True)


box.set_region((-xSize, xSize), (-ySize, ySize), (0, zSize), idx_mat2)
#box.set_region((-xSize, xSize), (-ySize, ySize), (zSize/2, zSize), idx_mat1)


detector = Detector(xdet,ydet,zdet,thetadet, phidet ,diametre,  NA ) #initialisation du detecteur
#f = open("photons_init.csv", "w")

# Coeur Montecarlo 

for inphoton in range (1,Nphoton+1):
	print(f"\rProgress: {inphoton / Nphoton * 100:.1f}%", end="", flush=True)
	#p = Photon(xinit,yinit,zinit,uxinit,uyinit,uzinit) # pencil beam
	#p = launch_plane_wave(zinit) #plan wave
	p = gaussian_source(xinit,yinit,zinit, W0, theta, zinit, uxinit, uyinit, uzinit) # source gaussienne
	
	box.getNewVoxelProperties(p) #photon prends les propriétés optiques de son voxel de départ
	
	#f.write(f"{p.x},{p.y},{p.z},{p.ux},{p.uy},{p.uz}\n")

	
	while (p.alive ):
		while (p.alive == True and p.stepLeft > 0):
			propagate(p) # propagation + absorption	
			if (p.sameVoxel == False):
				detect(p,detector) # si le photon s'est echappe par le cote du detecteur
				escape(p) # verifier frontiere en x,y bord (A) et z (R,T)
				#escape_periodic_xy(p) # comme escape mais periodique en x et y
				if (p.alive ): # si le photon ne s'est pas echappe alors 
					box.getNewVoxelProperties(p)  #  nouveau voxel donc potentiellement change de propriete
		if p.alive:
			roulette(p) # eleminer photon de faible energy ou leur laisser une chance de survie
		if p.alive:
			if (p.polar):
				scatter_polarized(p) # scattering du photon changement de ux uy uz  et de polarisation           
			else:
				scatter(p) # scattering du photon changement de ux uy uz


# ==========================================================
# BILAN GLOBAL DE LA SIMULATION
# ==========================================================

A_count = Nphoton - (compteurR + compteurT)

R_rate = compteurR / Nphoton
T_rate = compteurT / Nphoton
A_rate = 1.0 - R_rate - T_rate

C_rate = detector.collected / Nphoton

print("\n" + "="*66)
print("                 RÉSULTATS DE LA SIMULATION MONTE CARLO")
print("="*66)

print(f"\nNombre de photons émis : {Nphoton}")

# ----------------------------------------------------------
# Bilan global
# ----------------------------------------------------------
print("\n" + "-"*66)
print("BILAN GLOBAL")
print("-"*66)

print(f"{'Réflexion (R)':32s}: {compteurR:12.6f}   ({R_rate:10.6f})")
print(f"{'Transmission (T)':32s}: {compteurT:12.6f}   ({T_rate:10.6f})")
print(f"{'Absorption (A)':32s}: {A_count:12.6f}   ({A_rate:10.6f})")

# ----------------------------------------------------------
# Détecteur
# ----------------------------------------------------------
print("\n" + "-"*66)
print("DÉTECTEUR")
print("-"*66)

print(f"{'Photons collectés':32s}: {detector.collected:12.6f}")
print(f"{'Taux de collecte':32s}: {C_rate:12.6f}")
print(f"{'Pourcentage collecté':32s}: {100*C_rate:12.4f} %")

# ==========================================================
# POLARISATION
# ==========================================================
print("\n" + "="*66)
print("                     RÉSULTATS DE POLARISATION")
print("="*66)

# ----------------------------------------------------------
# Réflexion
# ----------------------------------------------------------
print("\n" + "-"*66)
print("RÉFLEXION")
print("-"*66)

print("Vecteur de Stokes brut :")
print(f"I = {R_I:.6f}")
print(f"Q = {R_Q:.6f}")
print(f"U = {R_U:.6f}")
print(f"V = {R_V:.6f}")

if R_I > 0:

    print("\nVecteur de Stokes normalisé :")
    print("I = 1.000000")
    print(f"Q = {R_Q/R_I:.6f}")
    print(f"U = {R_U/R_I:.6f}")
    print(f"V = {R_V/R_I:.6f}")

    DOP_R  = math.sqrt(R_Q**2 + R_U**2 + R_V**2) / R_I
    LDOP_R = math.sqrt(R_Q**2 + R_U**2) / R_I
    CDOP_R = abs(R_V) / R_I

    print("\nDegrés de polarisation :")
    print(f"Polarisation totale     = {DOP_R:.6f}")
    print(f"Polarisation linéaire  = {LDOP_R:.6f}")
    print(f"Polarisation circulaire= {CDOP_R:.6f}")

else:
    print("Aucun signal réfléchi")

# ----------------------------------------------------------
# Transmission
# ----------------------------------------------------------
print("\n" + "-"*66)
print("TRANSMISSION")
print("-"*66)

print("Vecteur de Stokes en nombre de photons :")
print(f"I = {T_I:.6f}")
print(f"Q = {T_Q:.6f}")
print(f"U = {T_U:.6f}")
print(f"V = {T_V:.6f}")

if T_I > 0:

    print("\nVecteur de Stokes normalisé :")
    print("I = 1.000000")
    print(f"Q = {T_Q/T_I:.6f}")
    print(f"U = {T_U/T_I:.6f}")
    print(f"V = {T_V/T_I:.6f}")

    DOP_T  = math.sqrt(T_Q**2 + T_U**2 + T_V**2) / T_I
    LDOP_T = math.sqrt(T_Q**2 + T_U**2) / T_I
    CDOP_T = abs(T_V) / T_I

    print("\nDegrés de polarisation :")
    print(f"Polarisation totale     = {DOP_T:.6f}")
    print(f"Polarisation linéaire  = {LDOP_T:.6f}")
    print(f"Polarisation circulaire= {CDOP_T:.6f}")

else:
    print("Aucun signal transmis")

# ----------------------------------------------------------
# Détecteur
# ----------------------------------------------------------
print("\n" + "-"*66)
print("DÉTECTEUR")
print("-"*66)

print("Vecteur de Stokes en nombre de photons :")
print(f"I = {detector.D_I:.6f}")
print(f"Q = {detector.D_Q:.6f}")
print(f"U = {detector.D_U:.6f}")
print(f"V = {detector.D_V:.6f}")

if detector.D_I > 0:

    print("\nVecteur de Stokes normalisé :")
    print("I = 1.000000")
    print(f"Q = {detector.D_Q/detector.D_I:.6f}")
    print(f"U = {detector.D_U/detector.D_I:.6f}")
    print(f"V = {detector.D_V/detector.D_I:.6f}")

    DOP_D  = math.sqrt(detector.D_Q**2 + detector.D_U**2 + detector.D_V**2) / detector.D_I
    LDOP_D = math.sqrt(detector.D_Q**2 + detector.D_U**2) / detector.D_I
    CDOP_D = abs(detector.D_V) / detector.D_I

    print("\nDegrés de polarisation :")
    print(f"Polarisation totale     = {DOP_D:.6f}")
    print(f"Polarisation linéaire  = {LDOP_D:.6f}")
    print(f"Polarisation circulaire= {CDOP_D:.6f}")

else:
    print("Aucun signal détecté")

print("\n" + "="*66)
print("Fin de simulation")
print("="*66)

# ==========================================================
