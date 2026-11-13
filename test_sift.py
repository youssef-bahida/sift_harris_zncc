# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE III : CALCUL DES DESCRIPTEURS SIFT
#
# Question 5 :
# Tester la methode SIFT dans un script qui appelle les fonctions
# precedentes pour une image donnee et visualiser le calcul
# des descripteurs SIFT pour les differents patchs.
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
    return (np.degrees(np.arctan2(Iy, Ix)) + 360) % 360

def gaussSIFT(I, sigma=8):
    h, w = I.shape
    x = np.arange(w) - (w - 1) / 2
    y = np.arange(h) - (h - 1) / 2
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))

def echantillonner(I, taille=16, pas=16):
    patches = []
    h, w = I.shape

    for y in range(0, h - taille + 1, pas):
        for x in range(0, w - taille + 1, pas):
            patches.append((I[y:y+taille, x:x+taille], (x, y)))

    return patches

def computeSIFT(P, Ig, Ior, Mg):
    vsift = np.zeros(128)
    bins = np.floor(Ior / 45).astype(int) % 8
    contributions = Ig * Mg

    for i in range(4):
        for j in range(4):
            y1, y2 = i * 4, (i + 1) * 4
            x1, x2 = j * 4, (j + 1) * 4

            hist = np.zeros(8)

            np.add.at(
                hist,
                bins[y1:y2, x1:x2].ravel(),
                contributions[y1:y2, x1:x2].ravel()
            )

            vsift[(i * 4 + j) * 8:(i * 4 + j + 1) * 8] = hist

    norme = np.linalg.norm(vsift)

    if norme > 0:
        vsift /= norme

    return vsift

def visuSIFT(I, Ig, Ior, patch, nom, s, sift):
    x, y = patch

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.gca().add_patch(
        plt.Rectangle((x, y), 16, 16, fill=False, edgecolor="red", linewidth=2)
    )
    plt.title(nom)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(sift)), sift)
    plt.title(f"Descripteur SIFT - Patch {s}")
    plt.xlabel("Dimension")
    plt.ylabel("Valeur")

    plt.tight_layout()
    plt.show()

# 1. Chargement de l'image
I = cv2.imread("DataTP3/image.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Image introuvable.")

# 2. Calcul des gradients et orientations
Ix, Iy, Ig = gradient(I)
Ior = orientation(Ix, Iy, Ig)

# 3. Extraction des patchs
patches = echantillonner(I)

print("Nombre de patchs :", len(patches))

# 4. Calcul et visualisation des descripteurs
for numero, (P, (x, y)) in enumerate(patches[:5], start=1):

    Ig_patch = Ig[y:y+16, x:x+16]
    Ior_patch = Ior[y:y+16, x:x+16]
    Mg = gaussSIFT(P)

    sift = computeSIFT(P, Ig_patch, Ior_patch, Mg)

    print(f"Patch {numero} - Coordonnees ({x}, {y})")
    print("Dimension du SIFT :", len(sift))

    visuSIFT(
        I, Ig, Ior,
        (x, y),
        "Visualisation des patchs SIFT",
        numero,
        sift
    )

print("Test SIFT termine.")
