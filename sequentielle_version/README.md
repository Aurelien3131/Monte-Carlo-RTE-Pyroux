Monte Carlo Radiative Transfer (RTE)
====================================

Simulation of light propagation in scattering and absorbing media using the Monte Carlo method.

This code implements a radiative transport solver for simulating photon propagation in turbid media (fog, aerosols, dust, particles media).

The model is based on a Monte Carlo approach to radiative transport, allowing the simulation of photon trajectories in a voxelized 3D medium with absorption, scattering, and detection.

Translated with DeepL.com (free version)
------------------------------------
Code Architecture
------------------------------------

main.py
  - Main simulation loop
  - Iterates over box thicknesses (Lz)
  - Iterates over wavelengths
  - Iterates over optical densities
  - Initializes the geometry, source, and detectors
  - Runs the Monte Carlo simulation
  - Saves the simulation results

config.py
  - Global parameters (physical, numerical, and geometrical)
  - Number of photons
  - Source and detector parameters

geometry.py
  - Definition of the 3D voxelized simulation domain
  - Material management
  - Detector definition

io_materials.py
  - Loading of optical properties
  - Phase function handling
  - Wavelength-dependent material data

propagation.py
  - Photon propagation
  - Voxel traversal
  - Absorption and boundary handling

scattering.py
  - Photon scattering
  - Phase function sampling
  - Photon direction updates

photon.py
  - Photon data structure
  - Position, direction, time, and weight
  - Polarization parameters (Stokes vector)

simulation.py
  - Main Monte Carlo simulation loop
  - Photon-medium interaction handling
  - Detector signal collection

sources.py
  - Photon generation
  - Plane, Gaussian, and modulated sources

param_laser.py
  - Laser parameters (beam waist, divergence, modulation)

output.py
  - Result export
  - Simulation logs and histograms

------------------------------------
Simulation Geometry
------------------------------------

- Simulation box aligned along the Z-axis
- Source located at z = 0
- Detector located at z = zSize

Simulation domain:
  x ∈ [-xSize, xSize]
  y ∈ [-ySize, ySize]
  z ∈ [0, zSize]

------------------------------------
Units
------------------------------------

Length: cm
Time: s
Optical coefficients (μa, μs): cm⁻¹
Angles: radians
Wavelength: µm

------------------------------------
Simulation Workflow
------------------------------------

For each simulation configuration:
- Box thickness (Lz)
- Wavelength
- Medium optical density
- Incident polarization state

The program:
1. Loads the optical properties of the medium.
2. Initializes the voxelized simulation geometry.
3. Initializes the source and detectors.
4. Runs the Monte Carlo simulation.
5. Saves the simulation results.

------------------------------------
Output Files (results directory)
------------------------------------

- Time-resolved histograms (main detector)
- Sub-detector histograms
- Emission histograms
- Simulation logs

------------------------------------
Authors
------------------------------------

Aurélien Roux & Julien Fades  
Fresnel Institute – Aix-Marseille University
