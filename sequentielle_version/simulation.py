# ==========================================================

# simulation.py

# Contains the main Monte Carlo simulation loop.
# Returns a dictionary containing all simulation results.
# ==========================================================


from config import (Nphoton, weightmin, chance, source_function, boundary_function)

from propagation import (propagate, escape, roulette, escape_periodic_xy)

from scattering import (scatter)

from polarization import (scatter_polarized)

from sources import gaussian_source, gaussian_source_modu, isotrope, plane_wave

from geometry import (Detector, Histogramme, HistogrammeEmission, detect)

from photon import Photon

def run_simulation(box, xSize, ySize, zSize, detector, sub_detector, histo, sub_histo, histo_emit, boundary_function, source_function, source_params, stokes_input):

    # Runs a complete Monte Carlo simulation.

    I_in, Q_in, U_in, V_in = stokes_input

    compteurR = 0.0
    compteurT = 0.0
    compteurA = 0.0

    R_I = R_Q = R_U = R_V = 0.0
    T_I = T_Q = T_U = T_V = 0.0

    for i in range(Nphoton):

        print(f"\rProgress : {(i+1)/Nphoton*100:5.1f} %", end="", flush=True)
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
                    detect(photon, sub_detector, sub_histo)
                    

                    match boundary_function: # choose the boundary conditions
                        case 1 :
                            status = escape(photon, xSize, ySize, zSize)
                        case 2 :                        
                            status = escape_periodic_xy(photon, zSize, xSize, ySize)                    
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

    print()

    compteurA = Nphoton - compteurR - compteurT

    return {

        "Nphoton": Nphoton,

        "Reflection": compteurR,
        "Transmission": compteurT,
        "Absorption": compteurA,

        "Detector": detector,
        "SubDetector": sub_detector,

        "Histogram": histo,
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
