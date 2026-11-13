# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE IV-A : IMPLEMENTATION DU DETECTEUR DE COINS DE HARRIS
#
# Question 1 :
# Calculer les images des gradients X et Y de l'image, puis
# filtrer ces images resultantes par un filtre gaussien.
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement de l'image
I = cv2.imread("DataTP3/image.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Image introuvable.")

# 2. Calcul des gradients X et Y avec Sobel
Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

# 3. Filtrage gaussien des gradients
Gx = cv2.GaussianBlur(Ix, (3, 3), 1)
Gy = cv2.GaussianBlur(Iy, (3, 3), 1)

# 4. Normalisation pour la visualisation
Ix_display = cv2.normalize(np.abs(Ix), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Iy_display = cv2.normalize(np.abs(Iy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Gx_display = cv2.normalize(np.abs(Gx), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Gy_display = cv2.normalize(np.abs(Gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 5. Affichage des resultats
images = [
    (I, "Image originale"),
    (Ix_display, "Gradient X - Sobel"),
    (Iy_display, "Gradient Y - Sobel"),
    (Gx_display, "Gradient X - Filtre gaussien"),
    (Gy_display, "Gradient Y - Filtre gaussien")
]

plt.figure(figsize=(15, 9))

for i, (img, titre) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap="gray")
    plt.title(titre)
    plt.axis("off")

plt.tight_layout()
plt.show()

# 6. Informations
print("Dimensions de l'image :", I.shape)
print("Gradient X :", Ix.shape)
print("Gradient Y :", Iy.shape)
print("Filtre gaussien : 3x3")
print("Sigma : 1")
