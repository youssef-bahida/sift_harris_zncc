# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE IV-A : IMPLEMENTATION DU DETECTEUR DE COINS DE HARRIS
#
# Question 3 :
# Calculer la matrice de Harris pour chaque pixel, sa trace,
# son determinant et la reponse R du detecteur :
# R = Det(M) - k * (Trace(M))^2
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement de l'image
I = cv2.imread("DataTP3/image.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Image introuvable.")

# 2. Conversion en float
I = np.float64(I)

# 3. Calcul des gradients
Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

# 4. Calcul des produits des gradients
Ix2 = Ix ** 2
Iy2 = Iy ** 2
Ixy = Ix * Iy

# 5. Filtrage gaussien des produits
M11 = cv2.GaussianBlur(Ix2, (3, 3), 1)
M22 = cv2.GaussianBlur(Iy2, (3, 3), 1)
M12 = cv2.GaussianBlur(Ixy, (3, 3), 1)

# 6. Calcul de la trace
trace = M11 + M22

# 7. Calcul du determinant
det = M11 * M22 - M12 ** 2

# 8. Calcul de la reponse de Harris
k = 0.04
R = det - k * (trace ** 2)

# 9. Visualisation
plt.figure(figsize=(14, 9))

plt.subplot(2, 2, 1)
plt.imshow(I, cmap="gray")
plt.title("Image originale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(trace, cmap="viridis")
plt.colorbar()
plt.title("Trace de la matrice M")

plt.subplot(2, 2, 3)
plt.imshow(det, cmap="viridis")
plt.colorbar()
plt.title("Determinant de M")

plt.subplot(2, 2, 4)
plt.imshow(R, cmap="hot")
plt.colorbar()
plt.title("Reponse de Harris R")

plt.tight_layout()
plt.show()

# 10. Informations
print("Parametre k :", k)
print("Trace maximale :", np.max(trace))
print("Determinant maximal :", np.max(det))
print("Reponse R maximale :", np.max(R))
print("Reponse R minimale :", np.min(R))
print("Nombre de pixels avec R positif :", np.sum(R > 0))
