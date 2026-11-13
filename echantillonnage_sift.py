# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE III : CALCUL DES DESCRIPTEURS SIFT
#
# Question 3 :
# Programmer la fonction echantillonner pour extraire des patchs
# de l'image. Chaque patch est de taille 16x16 pixels.
# Ces patchs seront ensuite utilises pour calculer les
# descripteurs SIFT avec computeSIFT et les visualiser avec visuSIFT.
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

def echantillonner(I, taille=16, pas=16):
    patches = []

    hauteur, largeur = I.shape[:2]

    for y in range(0, hauteur - taille + 1, pas):
        for x in range(0, largeur - taille + 1, pas):
            patch = I[y:y+taille, x:x+taille]
            patches.append((patch, (x, y)))

    return patches

# 1. Chargement de l'image
I = cv2.imread("DataTP3/image.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Image introuvable.")

# 2. Extraction des patchs
patches = echantillonner(I)

# 3. Affichage des informations
print("Dimensions de l'image :", I.shape)
print("Nombre total de patchs :", len(patches))

# 4. Visualisation des premiers patchs
nombre = min(16, len(patches))

plt.figure(figsize=(10, 10))

for i in range(nombre):
    patch, (x, y) = patches[i]

    plt.subplot(4, 4, i + 1)
    plt.imshow(patch, cmap="gray")
    plt.title(f"Patch {i+1} ({x},{y})")
    plt.axis("off")

plt.suptitle("Echantillonnage des patchs 16x16")
plt.tight_layout()
plt.show()
