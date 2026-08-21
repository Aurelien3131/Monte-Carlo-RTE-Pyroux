import numpy as np
import matplotlib.pyplot as plt

# Nom du fichier d'entrée
#fichier = "histo_small_10_v_40_z_5000_pol_1000_fog_less.txt"
fichier = "histo_small_10_v_40_l_9.8_z_5000_pol_1000.txt"

# Lecture des données
data = np.loadtxt(fichier, comments="#")

# Colonnes
temps = data[:, 0]
signal = data[:, 1]

# Pas temporel moyen
dt = np.mean(np.diff(temps))

# Nombre de points
N = len(signal)

# ==========================
# NORMALISATION
# ==========================

# Normalisation aire = 1
integrale = np.trapz(signal, temps)

if integrale != 0:
    signal = signal / integrale

# Retrait composante continue
signal = signal - np.mean(signal)

# Fenêtre de Hanning
fenetre = np.hanning(N)
signal = signal * fenetre

# ==========================
# FFT
# ==========================

fft_signal = np.fft.fft(signal)

frequences = np.fft.fftfreq(N, d=dt)

# Garder fréquences positives uniquement
mask = frequences >= 0

freq_pos = frequences[mask]
fft_pos = np.abs(fft_signal[mask])

# Normalisation FFT pour comparaison
fft_pos = fft_pos / np.max(fft_pos)

# ==========================
# ZOOM AUTOMATIQUE
# ==========================

# Seuil : garde les amplitudes > 1% du max
seuil = 0.01

indices_non_nuls = np.where(fft_pos > seuil)[0]

if len(indices_non_nuls) > 0:
    xmin = freq_pos[indices_non_nuls[0]]
    xmax = freq_pos[indices_non_nuls[-1]]
else:
    xmin = freq_pos[0]
    xmax = freq_pos[-1]

# Petite marge
marge = 0.05 * (xmax - xmin)

# ==========================
# TRACE
# ==========================

plt.figure(figsize=(10, 6))

plt.plot(freq_pos, fft_pos)

plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude normalisée")
plt.title("FFT du signal normalisé")

plt.grid(True)

# Zoom automatique
plt.xlim(
    max(0, xmin - marge),
    xmax + marge
)

plt.tight_layout()

# Sauvegarde
plt.savefig(
    "fft_spectre_fog_small_10_v_40_l_9.8.png",
    #"fft_spectre_fog_less_zoom.png",
    dpi=300,
    bbox_inches="tight"
)
# ==========================
# SAUVEGARDE FFT
# ==========================

# Colonnes :
# fréquence | amplitude FFT normalisée
fft_data = np.column_stack((freq_pos, fft_pos))

np.savetxt(
    "fft_fog_small_10_v_40_l_9.8.txt",
    #"fft_spectre_fog_less.txt",
    fft_data,
    header="Frequence_Hz\tAmplitude_FFT",
    fmt="%.8e",
    delimiter="\t"
)

print("FFT sauvegardée")
plt.show()
print("Graphique sauvegardé ")
