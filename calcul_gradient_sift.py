# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE III : CALCUL DES DESCRIPTEURS SIFT
#
# Question 1 :
# Charger une image de test I et ecrire une fonction gradient(I)
# pour calculer le module Ig du gradient en utilisant les masques
# de Sobel, ainsi que Ior = orientation(Ix, Iy, Ig) pour calculer
# l'orientation du gradient en chaque pixel.
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
    Ior = (Ior + 360) % 360
    return Ior

# 1. Chargement de l'image
I = cv2.imread("DataTP3/image.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Image introuvable.")

# 2. Calcul du gradient
Ix, Iy, Ig = gradient(I)

# 3. Calcul de l'orientation
Ior = orientation(Ix, Iy, Ig)

# 4. Normalisation pour la visualisation
Ig_display = cv2.normalize(Ig, None, 0, 255, cv2.NORM_MINMAX)

# 5. Affichage des resultats
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(I, cmap="gray")
plt.title("Image originale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(Ix, cmap="gray")
plt.title("Gradient horizontal Ix")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(Iy, cmap="gray")
plt.title("Gradient vertical Iy")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(Ig_display, cmap="gray")
plt.title("Module du gradient Ig")
plt.axis("off")

plt.tight_layout()
plt.show()

# 6. Affichage des informations
print("Dimensions de l'image :", I.shape)
print("Dimensions du module :", Ig.shape)
print("Dimensions de l'orientation :", Ior.shape)
print("Orientation minimale :", Ior.min())
print("Orientation maximale :", Ior.max())
