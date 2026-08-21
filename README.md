# Monte Carlo Radiative Transfer Equation Codes

This repository contains two Monte Carlo implementations for solving the **Radiative Transfer Equation (RTE)**:

* **Sequential version**: single-core implementation.
* **Parallel CPU version**: multi-core implementation.

The two versions are functionally identical. They differ only in the following files:

* `main.py`
* `simulation.py`

Each version includes its own `README` describing its specific architecture and usage.


-------------------------------------

WARNING : These codes do not take fluorescence or thermal effects into account.

WARNING 2 : For plane waves and pencil beams, there is a risk of freezing or inconsistent RTA (WIP)

-------------------------------------

If you want to change the settings for a simulation, simply edit config.py at minimum and main.py at most—particularly for the loops.

-------------------------------------

## Legacy Code

The old_MC_PYROUX directory contains previous development versions of the project. It is kept for archival purposes and is not required to run the current implementations.

## Example Applications

The examples folder contains a few applications that use this code. Note that these applications are not necessarily identical to the original code.

## Fog Data

The pavin_oct_alb folder contains experimental fog data, including DSD, μa, μs, and phase functions.

If you use these fog data, please cite:

Free-space optical transmission measurements from 0.532 to 10 µm in real controlled fog, https://doi.org/10.1364/OL.609352

-------------------------------------

## Laser Parameters

When using a **Gaussian laser source**, the beam parameters are specified in: param_laser.txt

Both the sequential and parallel implementations use this file.

-------------------------------------

## Output Directory

Simulation results are written to a directory named: result/ 

This folder is automatically used by both implementations to store the generated outputs (histograms, logs, detector data, etc.).

-------------------------------------

## Custom Fog Simulations

To simulate a custom fog, two types of input data are required:

1. **Optical properties** of the medium.
2. **Scattering phase function** of the fog.

In the provided example (`pavin_oct_alb` directory), the codes read:

* Optical coefficients from: coeff_optique_Pav_al_big_4_v_V_l_L_nm.txt


* Polarized phase function from the `ram` file.

-------------------------------------

## Polarized Phase Function

The `ram` file contains the scattering matrix elements computed from the fog **Droplet Size Distribution (DSD)** as a function of the scattering angle θ.

Its columns are organized as follows:

cos(θ)   S11(cos(θ))   S12(cos(θ))   S33(cos(θ))   S34(cos(θ))


where S11(cos(θ)) corresponds to the scattering phase function for the **unpolarized** case: S11(cos(θ)) = f(cos(θ))
-------------------------------------

The remaining Mueller matrix elements (`S12`, `S33`, and `S34`) describe the polarization properties of the scattering medium and are used during polarized Monte Carlo simulations.

-------------------------------------
Authors
-------------------------------------

Aurélien Roux & Julien Fades
Fresnel Institute – Aix-Marseille University
