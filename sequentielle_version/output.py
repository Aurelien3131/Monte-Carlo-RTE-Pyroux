# ======================================================================================================================

# output.py

# Saving and exporting simulation results.

# Contains:
# - writing emission time histograms
# - writing detector histograms
# - generation of simulation summary files
# - computation and export of polarization parameters
#   (Stokes, DOP, LDOP, CDOP)

# Output data are saved as text files
# suitable for post-processing and visualization.

# ======================================================================================================================


import math


# ==========================================================
# Histogram of the emitted photon
# ==========================================================

def write_emission_histogram(filename, histo_emit):

    with open(filename, "w", encoding="utf-8") as f:

        f.write("# Histogram of the emitted photon\n")
        f.write("# Time(s)\tSignal\n")

        centers, values = histo_emit.get()

        for t, v in zip(centers, values):
            f.write(f"{t:.12e}\t{v:.12e}\n")


# ==========================================================
# Histogram of the detector
# ==========================================================

def write_histogram(filename, histo, title="Temporal Histogram"):

    with open(filename, "w", encoding="utf-8") as f:

        f.write(f"# {title}\n")

        f.write("# Underflow\n")
        f.write(
            f"# temps = {histo.underflow_t:.12e} "
            f"poids = {histo.underflow:.12e} "
            f"nb = {histo.underflow_n}\n"
        )

        f.write("# Overflow\n")
        f.write(
            f"# temps = {histo.overflow_t:.12e} "
            f"poids = {histo.overflow:.12e} "
            f"nb = {histo.overflow_n}\n"
        )

        f.write("# Temps(s)\tSignal\tNombrePhotons\n")

        centers, values, number = histo.get()

        for t, v, n in zip(centers, values, number):

            f.write(f"{t:.12e}\t{v:.12e}\t{n:.12e}\n")


# ==========================================================
# Log 
# ==========================================================
def write_log(filename, results):

    Nphoton = results["Nphoton"]

    R = results["Reflection"]
    T = results["Transmission"]
    A = results["Absorption"]

    detector = results["Detector"]
    sub_detector = results["SubDetector"]

    R_I = results["R_I"]
    R_Q = results["R_Q"]
    R_U = results["R_U"]
    R_V = results["R_V"]

    T_I = results["T_I"]
    T_Q = results["T_Q"]
    T_U = results["T_U"]
    T_V = results["T_V"]

    with open(filename, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("MONTE CARLO SIMULATION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Number of photons : {Nphoton}\n\n")

        Rrate = R / Nphoton
        Trate = T / Nphoton
        Arate = A / Nphoton

        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")

        f.write(f"Reflection   : {R:.6f} ({Rrate:.6f})\n")
        f.write(f"Transmission : {T:.6f} ({Trate:.6f})\n")
        f.write(f"Absorption   : {A:.6f} ({Arate:.6f})\n\n")

        f.write("DETECTOR\n")
        f.write("-" * 70 + "\n")

        collect = detector.collected / Nphoton

        f.write(f"Collected photons : {detector.collected:.6f}\n")
        f.write(f"Collection rate   : {collect:.6f}\n")
        f.write(f"Percentage        : {100*collect:.4f} %\n\n")

        f.write("SUB-DETECTOR\n")
        f.write("-" * 70 + "\n")

        collect = sub_detector.collected / Nphoton

        f.write(f"Collected photons : {sub_detector.collected:.6f}\n")
        f.write(f"Collection rate   : {collect:.6f}\n")
        f.write(f"Percentage        : {100*collect:.4f} %\n\n")

        # =====================================================
        # REFLECTION POLARIZATION
        # =====================================================

        f.write("=" * 70 + "\n")
        f.write("REFLECTION POLARIZATION\n")
        f.write("=" * 70 + "\n")

        f.write(f"I = {R_I:.6f}\n")
        f.write(f"Q = {R_Q:.6f}\n")
        f.write(f"U = {R_U:.6f}\n")
        f.write(f"V = {R_V:.6f}\n\n")

        if R_I > 0:

            f.write("Normalized vector\n")

            f.write("I = 1.000000\n")
            f.write(f"Q = {R_Q/R_I:.6f}\n")
            f.write(f"U = {R_U/R_I:.6f}\n")
            f.write(f"V = {R_V/R_I:.6f}\n\n")

            dop = math.sqrt(R_Q**2 + R_U**2 + R_V**2) / R_I
            ldop = math.sqrt(R_Q**2 + R_U**2) / R_I
            cdop = abs(R_V) / R_I

            f.write(f"DOP  = {dop:.6f}\n")
            f.write(f"LDOP = {ldop:.6f}\n")
            f.write(f"CDOP = {cdop:.6f}\n\n")

        # =====================================================
        # TRANSMISSION POLARIZATION
        # =====================================================

        f.write("=" * 70 + "\n")
        f.write("TRANSMISSION POLARIZATION\n")
        f.write("=" * 70 + "\n")

        f.write(f"I = {T_I:.6f}\n")
        f.write(f"Q = {T_Q:.6f}\n")
        f.write(f"U = {T_U:.6f}\n")
        f.write(f"V = {T_V:.6f}\n\n")

        if T_I > 0:

            f.write("Normalized vector\n")

            f.write("I = 1.000000\n")
            f.write(f"Q = {T_Q/T_I:.6f}\n")
            f.write(f"U = {T_U/T_I:.6f}\n")
            f.write(f"V = {T_V/T_I:.6f}\n\n")

            dop = math.sqrt(T_Q**2 + T_U**2 + T_V**2) / T_I
            ldop = math.sqrt(T_Q**2 + T_U**2) / T_I
            cdop = abs(T_V) / T_I

            f.write(f"DOP  = {dop:.6f}\n")
            f.write(f"LDOP = {ldop:.6f}\n")
            f.write(f"CDOP = {cdop:.6f}\n\n")

        # =====================================================
        # DETECTOR POLARIZATION
        # =====================================================

        f.write("=" * 70 + "\n")
        f.write("DETECTOR POLARIZATION\n")
        f.write("=" * 70 + "\n")

        f.write(f"I = {detector.D_I:.6f}\n")
        f.write(f"Q = {detector.D_Q:.6f}\n")
        f.write(f"U = {detector.D_U:.6f}\n")
        f.write(f"V = {detector.D_V:.6f}\n\n")

        if detector.D_I > 0:

            f.write("Normalized vector\n")

            f.write("I = 1.000000\n")
            f.write(f"Q = {detector.D_Q/detector.D_I:.6f}\n")
            f.write(f"U = {detector.D_U/detector.D_I:.6f}\n")
            f.write(f"V = {detector.D_V/detector.D_I:.6f}\n\n")

            dop = math.sqrt(
                detector.D_Q**2 +
                detector.D_U**2 +
                detector.D_V**2
            ) / detector.D_I

            ldop = math.sqrt(
                detector.D_Q**2 +
                detector.D_U**2
            ) / detector.D_I

            cdop = abs(detector.D_V) / detector.D_I

            f.write(f"DOP  = {dop:.6f}\n")
            f.write(f"LDOP = {ldop:.6f}\n")
            f.write(f"CDOP = {cdop:.6f}\n")

