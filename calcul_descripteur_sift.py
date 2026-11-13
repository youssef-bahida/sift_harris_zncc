# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE III : CALCUL DES DESCRIPTEURS SIFT
#
# Question 4 :
# Mettre au point une fonction computeSIFT(P, Ig, Ior, Mg)
# qui calcule le descripteur SIFT d'un patch P de taille 16x16.
#
# Le patch est divise en 16 blocs de 4x4 pixels.
# Chaque bloc produit un histogramme de 8 bins d'orientation.
# Le descripteur final contient 16 x 8 = 128 valeurs.
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gradient(I):
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    Ig = np.sqrt(Ix**2 + Iy**2)
    return Ix, Iy, Ig

def orientation(Ix, Iy, Ig):
    Ior = np.degrees(np.arctan2(Iy, Ix))
    return (Ior + 360) % 360

def gaussSIFT(I, sigma=8):
    h, w = I.shape
    x = np.arange(w) - (w - 1) / 2
    y = np.arange(h) - (h - 1) / 2
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))

def computeSIFT(P, Ig, Ior, Mg):
    if P.shape != (16, 16):
        raise ValueError("Le patch doit etre de taille 16x16.")

    if Ig.shape != (16, 16) or Ior.shape != (16, 16) or Mg.shape != (16, 16):
        raise ValueError("Ig, Ior et Mg doivent etre de taille 16x16.")

    vsift = []

    for i in range(4):
        for j in range(4):

            y1, y2 = i * 4, (i + 1) * 4
            x1, x2 = j * 4, (j + 1) * 4

            magnitude = Ig[y1:y2, x1:x2]
            angles = Ior[y1:y2, x1:x2]
            weights = Mg[y1:y2, x1:x2]

            histogramme = np.zeros(8)

            bins = np.floor(angles / 45).astype(int) % 8

            contributions = magnitude * weights

            np.add.at(
                histogramme,
                bins.ravel(),
                contributions.ravel()
            )

            vsift.extend(histogramme)

    vsift = np.array(vsift, dtype=np.float64)

    # Normalisation L2
    norme = np.linalg.norm(vsift)

    if norme > 0:
        vsift = vsift / norme

    return vsift

# 1. Chargement de l'image
I = cv2.imread("DataTP3/image.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Image introuvable.")

# 2. Calcul des gradients
Ix, Iy, Ig = gradient(I)

# 3. Calcul des orientations
Ior = orientation(Ix, Iy, Ig)

# 4. Extraction du premier patch valide
if I.shape[0] < 16 or I.shape[1] < 16:
    raise ValueError("Image trop petite.")

x, y = 0, 0
P = I[y:y+16, x:x+16]
Ig_patch = Ig[y:y+16, x:x+16]
Ior_patch = Ior[y:y+16, x:x+16]

# 5. Creation du masque gaussien
Mg = gaussSIFT(P)

# 6. Calcul du descripteur SIFT
vsift = computeSIFT(P, Ig_patch, Ior_patch, Mg)

# 7. Affichage des resultats
print("Taille du patch :", P.shape)
print("Taille du masque :", Mg.shape)
print("Dimension du descripteur :", len(vsift))
print("Descripteur SIFT :")
print(np.round(vsift, 4))

# 8. Visualisation du descripteur
plt.figure(figsize=(12, 4))
plt.bar(range(128), vsift)
plt.title("Descripteur SIFT - 128 dimensions")
plt.xlabel("Indice du descripteur")
plt.ylabel("Valeur")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
