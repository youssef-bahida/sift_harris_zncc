# TP3 - Détection et mise en correspondance de points d'intérêt
# IV-A.5 : Extraction des N meilleurs points de Harris par tri par insertion.
# N est un paramètre configurable (valeur par défaut : 50).

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE = "DataTP3/image.jpg"
N = 50
K = 0.04

image = cv2.imread(IMAGE)

if image is None:
    raise FileNotFoundError(f"Image introuvable : {IMAGE}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

M11 = cv2.GaussianBlur(Ix * Ix, (3, 3), 1)
M22 = cv2.GaussianBlur(Iy * Iy, (3, 3), 1)
M12 = cv2.GaussianBlur(Ix * Iy, (3, 3), 1)

trace = M11 + M22
det = M11 * M22 - M12 ** 2
R = det - K * trace ** 2

# Détection des maxima locaux strictement supérieurs aux 8 voisins
maxima = np.zeros(R.shape, dtype=bool)
center = R[1:-1, 1:-1]

neighbors = np.stack([
    R[:-2, :-2], R[:-2, 1:-1], R[:-2, 2:],
    R[1:-1, :-2],                 R[1:-1, 2:],
    R[2:, :-2], R[2:, 1:-1], R[2:, 2:]
], axis=-1)

maxima[1:-1, 1:-1] = (center > 0) & np.all(
    center[..., None] > neighbors, axis=-1
)

ys, xs = np.where(maxima)

# Tableau de taille N trié par ordre décroissant des réponses
best_scores = np.full(N, -np.inf)
best_x = np.full(N, -1, dtype=int)
best_y = np.full(N, -1, dtype=int)

# Tri par insertion
for y, x in zip(ys, xs):
    score = R[y, x]

    for i in range(N):
        if score > best_scores[i]:
            best_scores[i + 1:] = best_scores[i:-1]
            best_x[i + 1:] = best_x[i:-1]
            best_y[i + 1:] = best_y[i:-1]

            best_scores[i] = score
            best_x[i] = x
            best_y[i] = y
            break

# Affichage des N meilleurs points
result = image.copy()
valid = best_x >= 0

for rank, (x, y) in enumerate(zip(best_x[valid], best_y[valid]), 1):
    cv2.circle(result, (x, y), 4, (0, 0, 255), 1)
    cv2.putText(result, str(rank), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

print(f"Nombre de maxima locaux positifs : {len(xs)}")
print(f"Nombre de points sélectionnés : {np.sum(valid)}")
print("\nClassement des meilleurs points :")

for rank, (x, y, score) in enumerate(
    zip(best_x[valid], best_y[valid], best_scores[valid]), 1
):
    print(f"{rank:02d}. Coordonnées = ({x}, {y}) | Réponse Harris = {score:.2f}")

plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f"Les {np.sum(valid)} meilleurs points de Harris")
plt.axis("off")
plt.show()
