# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE III : CALCUL DES DESCRIPTEURS SIFT
#
# Question 2 :
# Ecrire une fonction Mg = gaussSIFT(I) pour calculer le masque
# gaussien utilise pour ponderer le calcul du descripteur SIFT.
# Le masque est de taille 16x16.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def gaussSIFT(I, sigma=8):
    h, w = I.shape[:2]

    x = np.arange(w) - (w - 1) / 2
    y = np.arange(h) - (h - 1) / 2

    X, Y = np.meshgrid(x, y)

    Mg = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    return Mg

# 1. Creation d'un patch de test 16x16
I = np.zeros((16, 16))

# 2. Calcul du masque gaussien
Mg = gaussSIFT(I)

# 3. Affichage du masque
plt.figure(figsize=(7, 6))
plt.imshow(Mg, cmap="hot")
plt.colorbar(label="Poids gaussien")
plt.title("Masque gaussien SIFT 16x16")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# 4. Affichage des valeurs
print("Dimensions du masque :", Mg.shape)
print("Poids maximal :", Mg.max())
print("Poids minimal :", Mg.min())
print("Matrice gaussienne :")
print(np.round(Mg, 3))
