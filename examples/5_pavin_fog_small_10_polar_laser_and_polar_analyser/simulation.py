# ======================================================================================================================

# simulation.py

# Main simulation loop for parallel and sequential Monte Carlo.

# Contains:
# - simulation_worker : Monte Carlo execution of a batch of photons
# - merge_results : aggregation of multi-process results
# - run_simulation : sequential or parallel orchestration

# The simulation follows photon transport in a voxelized medium:
# - free propagation
# - optical interactions (scattering, absorption)
# - polarization (optional)
# - detection (sensors and time histograms)

# Parallel version:
# Each process handles a subset of photons.
# Results are then merged in the main process.


# ======================================================================================================================

from multiprocessing import Pool
from config import NPROC, Nphoton, weightmin, chance, source_function, boundary_function
from propagation import propagate, escape,escape_periodic_xy, roulette
from scattering import scatter
from polarization import scatter_polarized
from sources import gaussian_source, gaussian_source_modu , isotrope, plane_wave
from geometry import (Detector, Histogramme, HistogrammeEmission, detect, detect_polar)
from photon import Photon

import random
base_seed = random.randint(0, 2**32 - 1)

# for display # important for visually seeing when code crashes
import time
spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

def simulation_worker( nphotons, box, xSize, ySize, zSize, xdet, ydet, zdet, thetadet, phidet, diametre, NA, t_ballistic, time_gate, time_window, bins, freq, nb_period, boundary_function, source_function, source_params, stokes_input, stokes_det, worker_id=None, worker_seed=None):

    random.seed(worker_seed) # ensures different random numbers for each worker

    I_in, Q_in, U_in, V_in = stokes_input
    I_det, Q_det, U_det, V_det = stokes_det	

    # ==========================================================
    # Worker-local objects
    # ==========================================================

    detector = Detector(xdet, ydet, zdet, thetadet, phidet, diametre, NA)
    detector_polar = Detector(xdet, ydet, zdet, thetadet, phidet, diametre, NA)
    sub_detector = Detector(xdet, ydet, zdet, thetadet, phidet, diametre, 0.5)
    histo_emit = HistogrammeEmission(time_in=0, time_out=nb_period / freq, time_bin=bins)
    histo = Histogramme(time_in=t_ballistic - time_gate, time_out=t_ballistic + time_window, time_bin=bins)
    histo_polar = Histogramme(time_in=t_ballistic - time_gate, time_out=t_ballistic + time_window, time_bin=bins)
    sub_histo = Histogramme(time_in=t_ballistic - time_gate, time_out=t_ballistic + time_window, time_bin=bins,)

    compteurR = 0.0
    compteurT = 0.0
    compteurA = 0.0

    R_I = R_Q = R_U = R_V = 0.0
    T_I = T_Q = T_U = T_V = 0.0

    for i in range(nphotons):
        if NPROC == 1 :
            print( f"\rProgress : {(i+1)/nphotons*100:5.1f} %", end="", flush=True)
        if NPROC > 1:
            if i % 5000 == 0:   # ultra faible fréquence
                s = spinner[(i // 5000 + worker_id) % len(spinner)]
                print(f"\r {s} ",end="", flush=True)
                
        match source_function: # choose the light source
            case 0 :
                photon = Photon(*source_params)
            case 1 : 
                photon = gaussian_source_modu(*source_params)
            case 2 : 
                photon = isotrope(*source_params)
            case 3 : 
                photon = gaussian_source(*source_params)
            case 4 : 
                photon = plane_wave(*source_params)               
                
                
        photon.I = I_in
        photon.Q = Q_in
        photon.U = U_in
        photon.V = V_in

        histo_emit.add_emit(photon)

        box.getNewVoxelProperties(photon)

        while photon.alive:

            while photon.alive and photon.stepLeft > 0:

                propagate(photon, box)

                if not photon.sameVoxel:

                    detect(photon, detector, histo)
                    detect_polar(photon, detector_polar, histo_polar, I_det, Q_det, U_det, V_det)
                    detect(photon, sub_detector, sub_histo)
                    
                    match boundary_function: # choose the boundary conditions
                        case 1 :
                            status = escape(photon, xSize, ySize, zSize)
                        case 2 :
                            status = escape_periodic_xy(photon,zSize,xSize,ySize)

                    if status == "R":

                        compteurR += photon.weight

                        R_I += photon.I * photon.weight
                        R_Q += photon.Q * photon.weight
                        R_U += photon.U * photon.weight
                        R_V += photon.V * photon.weight

                    elif status == "T":

                        compteurT += photon.weight

                        T_I += photon.I * photon.weight
                        T_Q += photon.Q * photon.weight
                        T_U += photon.U * photon.weight
                        T_V += photon.V * photon.weight

                    elif status in ("X", "Y"):

                        compteurA += photon.weight

                    if photon.alive:
                        box.getNewVoxelProperties(photon)

            if photon.alive:

                status = roulette(photon, weightmin, chance)

                if status == "dead":
                    compteurA += photon.weight

            if photon.alive:

                if photon.polar:
                    scatter_polarized(photon)
                else:
                    scatter(photon)


    compteurA = nphotons - compteurR - compteurT

    return {

        "Nphoton": nphotons,

        "Reflection": compteurR,
        "Transmission": compteurT,
        "Absorption": compteurA,

        "Detector": detector,
        "Detector_polar": detector_polar,
        "SubDetector": sub_detector,

        "Histogram": histo,
        "Histogram_polar": histo_polar,
        "SubHistogram": sub_histo,
        "EmissionHistogram": histo_emit,

        "R_I": R_I,
        "R_Q": R_Q,
        "R_U": R_U,
        "R_V": R_V,

        "T_I": T_I,
        "T_Q": T_Q,
        "T_U": T_U,
        "T_V": T_V,
    }



	
		
def merge_results(results):

    merged = results[0]

    for r in results[1:]:

        merged["Reflection"] += r["Reflection"]
        merged["Transmission"] += r["Transmission"]
        merged["Absorption"] += r["Absorption"]

        merged["Detector"].merge(r["Detector"])
        merged["SubDetector"].merge(r["SubDetector"])

        merged["Histogram"].merge(r["Histogram"])
        merged["Histogram_polar"].merge(r["Histogram_polar"])
        merged["SubHistogram"].merge(r["SubHistogram"])
        merged["EmissionHistogram"].merge(r["EmissionHistogram"])

        merged["R_I"] += r["R_I"]
        merged["R_Q"] += r["R_Q"]
        merged["R_U"] += r["R_U"]
        merged["R_V"] += r["R_V"]

        merged["T_I"] += r["T_I"]
        merged["T_Q"] += r["T_Q"]
        merged["T_U"] += r["T_U"]
        merged["T_V"] += r["T_V"]

        merged["Nphoton"] += r["Nphoton"]

    return merged
    
    
def run_simulation(box, xSize, ySize, zSize, xdet, ydet, zdet, thetadet, phidet, diametre, NA, t_ballistic, time_gate, time_window, bins, freq, nb_period, boundary_function, source_function, source_params, stokes_input, stokes_det):

    # -----------------------------
    # Sequential version
    # -----------------------------
    if NPROC == 1:

        return simulation_worker( Nphoton, box, xSize, ySize, zSize, xdet, ydet, zdet, thetadet, phidet, diametre, NA, t_ballistic, time_gate, time_window, bins, freq, nb_period, boundary_function, source_function, source_params, stokes_input, stokes_det)

    # -----------------------------
    # Parallel version
    # -----------------------------

    photons = [Nphoton // NPROC] * NPROC
    photons[-1] += Nphoton % NPROC

    tasks = []

    for i,n in enumerate(photons):
        worker_seed = base_seed + i
        tasks.append((n, box, xSize, ySize, zSize, xdet, ydet, zdet, thetadet, phidet, diametre, NA, t_ballistic, time_gate, time_window, bins, freq, nb_period, boundary_function, source_function, source_params, stokes_input, stokes_det, i, worker_seed))
     # i = worker_id # worker_seed  = seed random seed for worker

    with Pool(processes=NPROC) as pool:
        results = pool.starmap(simulation_worker, tasks)

    return merge_results(results)
