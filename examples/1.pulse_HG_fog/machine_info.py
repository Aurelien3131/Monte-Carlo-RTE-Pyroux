#======================================================================================================================

# machine_info.py

# Machine configuration diagnostics for preparing and optimizing
# the parallel execution of Monte Carlo simulations.

# Contains:
# - operating system and architecture detection
# - CPU analysis (physical and logical cores)
# - available RAM estimation
# - recommendation of the number of processes to use

#======================================================================================================================
import os
import platform
import psutil
import multiprocessing

def infos_pc():
    print("=" * 50)
    print("INFORMATIONS MACHINE")
    print("=" * 50)

    print(f"Système       : {platform.system()} {platform.release()}")
    print(f"Version       : {platform.version()}")
    print(f"Machine       : {platform.machine()}")
    print(f"Processeur    : {platform.processor()}")

    print("\nCPU")
    print("-" * 50)

    cpu_physiques = psutil.cpu_count(logical=False)
    cpu_logiques = psutil.cpu_count(logical=True)

    print(f"Cœurs physiques : {cpu_physiques}")
    print(f"Threads logiques: {cpu_logiques}")

    freq = psutil.cpu_freq()
    if freq:
        print(f"Fréquence actuelle : {freq.current:.0f} MHz")
        print(f"Fréquence max      : {freq.max:.0f} MHz")

    print(f"Charge CPU actuelle: {psutil.cpu_percent(interval=1)} %")

    print("\nRAM")
    print("-" * 50)

    ram = psutil.virtual_memory()
    print(f"RAM totale : {ram.total / (1024**3):.2f} Go")
    print(f"RAM libre  : {ram.available / (1024**3):.2f} Go")

    print("\nCONSEIL PARALLÉLISATION")
    print("-" * 50)

    # Recommandation prudente
    recommandation = max(1, cpu_logiques - 1)

    print(f"Nombre max théorique de workers : {cpu_logiques}")
    print(f"Nombre conseillé à tester       : {recommandation}")

    print("\nExemple multiprocessing :")
    print(f"Pool(processes={recommandation})")

if __name__ == "__main__":
    infos_pc()
