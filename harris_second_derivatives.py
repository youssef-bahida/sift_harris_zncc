# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE IV-A : IMPLEMENTATION DU DETECTEUR DE COINS DE HARRIS
#
# Question 2 :
# Calculer les gradients XX et YY de l'image en utilisant
# deux fois de suite l'operateur derive, puis filtrer
# les images obtenues par un filtre gaussien.
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement de l'image
I = cv2.imread("DataTP3/image.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Image introuvable.")

# 2. Premier calcul des gradients
Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

# 3. Calcul des derivees secondes
Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)

# 4. Filtrage gaussien
Gxx = cv2.GaussianBlur(Ixx, (3, 3), 1)
Gyy = cv2.GaussianBlur(Iyy, (3, 3), 1)

# 5. Normalisation pour affichage
def normaliser(img):
    return cv2.normalize(
        np.abs(img), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

# 6. Visualisation
images = [
    (I, "Image originale"),
    (normaliser(Ixx), "Derivee seconde Ixx"),
    (normaliser(Iyy), "Derivee seconde Iyy"),
    (normaliser(Gxx), "Ixx apres filtrage gaussien"),
    (normaliser(Gyy), "Iyy apres filtrage gaussien")
]

plt.figure(figsize=(15, 9))

for i, (img, titre) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap="gray")
    plt.title(titre)
    plt.axis("off")

plt.tight_layout()
plt.show()

# 7. Informations
print("Dimensions de Ixx :", Ixx.shape)
print("Dimensions de Iyy :", Iyy.shape)
print("Dimensions de Gxx :", Gxx.shape)
print("Dimensions de Gyy :", Gyy.shape)
print("Filtre gaussien : 3x3, sigma=1")
