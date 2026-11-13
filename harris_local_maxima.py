# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE IV-A : IMPLEMENTATION DU DETECTEUR DE COINS DE HARRIS
#
# Question 4 :
# Extraire les maxima locaux positifs dans un voisinage 3x3
# de la reponse R, en mettant a zero les points negatifs
# ou non superieurs a leurs 8 voisins.
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement de l'image
I = cv2.imread("DataTP3/image.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Image introuvable.")

I = np.float64(I)

# 2. Calcul des gradients
Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

# 3. Calcul des produits des gradients
Ix2 = cv2.GaussianBlur(Ix**2, (3, 3), 1)
Iy2 = cv2.GaussianBlur(Iy**2, (3, 3), 1)
Ixy = cv2.GaussianBlur(Ix * Iy, (3, 3), 1)

# 4. Calcul de la reponse de Harris
trace = Ix2 + Iy2
det = Ix2 * Iy2 - Ixy**2
k = 0.04
R = det - k * trace**2

# 5. Extraction des maxima locaux 3x3
voisinage = np.ones((3, 3), dtype=np.uint8)
maximum_local = cv2.dilate(R, voisinage)

masque = (R > 0) & (R == maximum_local)

# 6. Conservation des maxima
R_max = np.zeros_like(R)
R_max[masque] = R[masque]

# 7. Affichage des points detectes
resultat = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_GRAY2BGR)

y, x = np.where(masque)

for px, py in zip(x, y):
    cv2.circle(resultat, (px, py), 2, (0, 0, 255), -1)

# 8. Visualisation
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.imshow(I, cmap="gray")
plt.title("Image originale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(R, cmap="hot")
plt.title("Reponse de Harris R")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB))
plt.title("Maxima locaux positifs")
plt.axis("off")

plt.tight_layout()
plt.show()

# 9. Informations
print("Nombre de maxima locaux positifs :", len(x))
print("Dimensions de R :", R.shape)
