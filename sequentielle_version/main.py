# ======================================================================================================================
#                                    MONTE CARLO RADIATIVE TRANSFER (RTE)
#                               Light propagation in turbid media
#
#                                        © Roux Aurélien & Julien Fades
#                                              All rights reserved
#
# This code is a Monte Carlo implementation of the Radiative Transfer Equation (RTE)
# for simulating photon propagation in scattering and absorbing media
# (fog, aerosols, particulate media, etc.), mainly inspired by MCMatlab
# and the Ramella code.
#
# Owner: Roux Aurélien / Fresnel Institute / Aix-Marseille University
# Creation date: 06/23/2026
# Language: Python 3
#
# This program is intended for scientific or academic use.
#
# ======================================================================================================================
# GENERAL CODE ARCHITECTURE
# ======================================================================================================================
#
# main.py
# ├── Main simulation loop
# │   ├── Loop over box thicknesses (Lz)
# │   ├── Loop over wavelengths
# │   ├── Loop over particle densities / optical properties
# │   ├── Geometry initialization
# │   ├── Source / detector initialization
# │   ├── Monte Carlo execution
# │   └── Save results
# │
# ├── config.py
# │      Global parameters:
# │      - physical constants
# │      - numerical parameters
# │      - number of photons
# │      - geometry
# │      - source/detector parameters
# │
# ├── geometry.py
# │      Domain definition:
# │      - simulation box
# │      - voxelization
# │      - detectors
# │      - histograms
# │      - detection functions
# │
# ├── io_materials.py
# │      Input file reader:
# │      - optical coefficients
# │      - phase functions
# │      - polarization matrices
# │
# ├── output.py
# │      Output routines:
# │      - logs
# │      - histograms
# │      - final results
# │
# ├── param_laser.py
# │      Laser source parameters:
# │      - beam waist
# │      - divergence
# │      - temporal modulation
# │
# ├── photon.py
# │      Photon structure:
# │      - position
# │      - direction
# │      - time
# │      - weight
# │      - Stokes parameters
# │
# ├── propagation.py
# │      Photon propagation:
# │      - photon displacement
# │      - voxel crossing
# │      - absorption
# │      - boundary handling
# │
# ├── scattering.py
# │      Scattering:
# │      - phase function
# │      - polarized scattering
# │      - direction update
# │
# ├── simulation.py
# │      Monte Carlo core:
# │      - photon loop
# │      - medium interaction
# │      - detector collection
# │      - global statistics
# │
# └── sources.py
#        Photon generation:
#        - plane wave
#        - Gaussian beam
#        - modulated emission
#
#
# ======================================================================================================================
# SIMULATION GEOMETRY
# ======================================================================================================================
#
# Current convention:
#
# • The simulation box is oriented along the Z-axis.
# • The source is located on the input face:
#
#           z = 0
#
# • The detector is located on the output face:
#
#           z = zSize
#
#
#
#                         +Y
#                         ↑
#                         │
#                         │
#                         │
#
#               ┌──────────────────────┐
#              /                      /│
#             /                      / │
#            /                      /  │
#           └──────────────────────┘   │
#           │      TURBID MEDIUM   │   │
#           │                      │   │
#           │                      │   │
#           │                      │   │
#           │                      │   │
#           │                      │  /
#           │                      │/
#           └──────────────────────┘
#
#      Source                          Detector
#      (z = 0)                        (z = zSize)
#
#         ●──────────────────────────────►
#                     +Z
#
#
# Origin:
#
#                 (0,0,0)
#
# Simulation domain:
#
#     x ∈ [-xSize , +xSize]
#     y ∈ [-ySize , +ySize]
#     z ∈ [0      , zSize ]
#
#
# ======================================================================================================================
# UNITS
# ======================================================================================================================
#
# Length                → cm
# Time                  → s
# Optical coefficients  → cm⁻¹ (μa, μs)
# Angles                → radians
# Wavelength            → µm
#
# ======================================================================================================================

from config import * 
from geometry import * 
from io_materials import * 
from simulation import run_simulation
from output import *
import os

for zSize in LZ_values: #  Loop over the simulation box thicknesses (Lz)
    zdet = zSize # Detector z-position in the simulation box, can be config in config.py
    box = Box(xSize, ySize, zSize, Nx, Ny, Nz) # Initialisation of the simulation box
    dz = zSize/Nz
    for lambada in Lambda_values: # Loop over wavelength
        for dense in density_values: # Loop over fog densities (the phase-function filename and optical parameters depend on density and wavelength).
        # ====================================================================================================================
        # Media parameters

            coeff_file = "pavin_oct_alb/coeff_potique_Pav_al_small_10_v_"+str(dense)+"_l_"+str(lambada)+"_nm.txt" # Read mua mus g mut 
            mua_file, mut_file, mus_file, g_file = load_optical_coeffs(coeff_file)
            # WARNING mua et mus must be in cm^-1

            # Custom fog unpolar
            idx_mat2 = box.add_material(mua= mua_file, mus= mus_file, g=None, n=1, phase_file="pavin_oct_alb/ram_function_phases_Pav_al_small_10_v_"+str(dense)+"_l_"+str(lambada)+"_nm.txt", polar=False) 
            
            # Custom fog polar
            #idx_mat1 = box.add_material(mua= mua_file, mus= mus_file, g=None, n=1, phase_file="pavin_oct_alb/ram_function_phases_Pav_al_small_10_v_"+str(dense)+"_l_"+str(lambada)+"_nm.txt", polar=True) 
            
            # Henyey-Greenstein fog,  mua et mus in cm^-1 
            #idx_mat3 = box.add_material(mua= 0.001, mus= 50, g=0.8, n=1, phase_file=None)     
            
                   
            # If one media
            box.set_region((-xSize, xSize), (-ySize, ySize), (0, zSize), idx_mat2)
            
             
            # If two medias
            #box.set_region((-xSize, xSize), (-ySize, ySize), (0, zSize/2), idx_mat1)          
            #box.set_region((-xSize, xSize), (-ySize, ySize), (zSize/2, zSize), idx_mat2)             
        # ====================================================================================================================

            t_ballistic = (zdet-zinit)/ (100*C_VELOCITY/1.0) # Factor of 100 because c is expressed in m/s while z is in cm; 1.0 is the refractive index of the medium.
            #can be config in config.py
            # t_ballistic: minimum time required for a photon to travel ballistically from the source to the detector through the simulation domain. Used to define the time histogram.

            for pol_name, stokes in polar_inputs.items(): # Loop over initials polarizations
                print("LZ =", zSize, " POL =", pol_name, "lambda =", lambada, "visibilite", dense)
                I_in, Q_in, U_in, V_in = stokes


		# ==========================================================
                # INITIALISATION
                # ==========================================================

                # Photon counters initialization
                compteurR = compteurT = compteurA1 = 0
                R_I = R_Q = R_U = R_V = 0.0
                T_I = T_Q = T_U = T_V = 0.0
                
                # Detector initialisation
                detector = Detector(xdet,ydet,zdet,thetadet, phidet ,diametre,  NA ) 
                
                # Sub-Detector initialisation
                sub_detector = Detector(xdet,ydet,zdet,thetadet, phidet ,diametre,  0.05) 

        	# Temporal histogramms initialisation  of the emitted photons                      
                histo_emit = HistogrammeEmission(time_in=0,    time_out=nb_period/freq,    time_bin=bins) 
                
        	# Temporal histogramms initialisation  of the detector                                         
                histo = Histogramme(time_in = t_ballistic - time_gate  , time_out = t_ballistic + time_window , time_bin=bins)
                
        	# Temporal histogramms initialisation  of the sub-detector                                                         
                sub_histo = Histogramme(time_in = t_ballistic - time_gate  , time_out = t_ballistic + time_window , time_bin=bins)

                # gaussian source
                w0, _ = laser_params(lambada)      # w0
                _, theta = laser_params(lambada)   # theta
                source_params = (xinit, yinit, zinit, dz, w0, theta, zinit, uxinit, uyinit, uzinit, freq, nb_period, fact_modu)
                
                # ==========================================================
                # MONTECARLO CORE
                # ==========================================================
                
                results = run_simulation(box, xSize, ySize, zSize, detector, sub_detector, histo, sub_histo, histo_emit, boundary_function, source_function, source_params, stokes) 
                
                # ==========================================================
                # GLOBAL RESULT OF THE SIMULATION
                # ==========================================================
        
                # Create the “result” folder if it does not exist
                os.makedirs("result", exist_ok=True)
        
                nom_fichier = ("result/log_small_10_freq_"+str(freq)+"_v_"+str(dense)+"_l_"+str(lambada)+"_z_" + str(zSize) + "_pol_" + str(pol_name) + ".txt")
                nom_histo = ("result/histo_small_10_freq_"+str(freq)+"_v_"+str(dense)+"_l_"+str(lambada)+"_z_" + str(zSize) + "_pol_" + str(pol_name) + ".txt")
                nom_sub_histo = ("result/sub_histo_small_10_freq_"+str(freq)+"_v_"+str(dense)+"_l_"+str(lambada)+"_z_" + str(zSize) + "_pol_" + str(pol_name) + ".txt")
                nom_emit_histo = ("result/histo_emit_small_10_freq_"+str(freq)+"_v_"+str(dense)+"_l_"+str(lambada)+"_z_" + str(zSize) + "_pol_" + str(pol_name) + ".txt")

                write_emission_histogram(nom_emit_histo, histo_emit)
                write_histogram(nom_histo, histo, title="Histogramme temporel")
                write_histogram(nom_sub_histo, sub_histo, title="Sous Detecteur Histogramme temporel")
                write_log( nom_fichier, results)










