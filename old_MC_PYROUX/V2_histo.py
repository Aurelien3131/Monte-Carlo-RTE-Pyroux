import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier
data = pd.read_csv("photons_init.csv")

# Vérifier les colonnes
print("Colonnes :", data.columns)

# Nombre de colonnes
cols = data.columns
n = len(cols)

# Création des histogrammes
for col in cols:
    plt.figure()
    plt.hist(data[col], bins=100)
    plt.title(f"Histogramme de {col}")
    plt.xlabel(col)
    plt.ylabel("Nombre d'échantillons")

plt.show()
