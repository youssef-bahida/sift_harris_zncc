# TP3 - Détection et mise en correspondance de points d'intérêt
# IV-B.1 : Mise en correspondance de deux images par ZNCC.
# Pour chaque point de Harris de la première image, rechercher
# le point le plus ressemblant dans la deuxième image.

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE1 = "DataTP3/tem0.jpg"
IMAGE2 = "DataTP3/tem1.jpg"

N = 50
K = 11
HARRIS_K = 0.04

def detecter_harris(image, n):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    M11 = cv2.GaussianBlur(Ix * Ix, (3, 3), 1)
    M22 = cv2.GaussianBlur(Iy * Iy, (3, 3), 1)
    M12 = cv2.GaussianBlur(Ix * Iy, (3, 3), 1)

    trace = M11 + M22
    det = M11 * M22 - M12 ** 2
    R = det - HARRIS_K * trace ** 2

    center = R[1:-1, 1:-1]
    neighbors = np.stack([
        R[:-2, :-2], R[:-2, 1:-1], R[:-2, 2:],
        R[1:-1, :-2], R[1:-1, 2:],
        R[2:, :-2], R[2:, 1:-1], R[2:, 2:]
    ], axis=-1)

    maxima = np.zeros(R.shape, dtype=bool)
    maxima[1:-1, 1:-1] = (
        (center > 0) & np.all(center[..., None] > neighbors, axis=-1)
    )

    ys, xs = np.where(maxima)
    points = sorted(
        [(x, y, R[y, x]) for y, x in zip(ys, xs)],
        key=lambda p: p[2],
        reverse=True
    )

    return [(x, y) for x, y, _ in points[:n]]

def calculer_zncc(patch1, patch2):
    A = patch1.astype(np.float64)
    B = patch2.astype(np.float64)

    A -= A.mean()
    B -= B.mean()

    denominateur = np.sqrt(np.sum(A ** 2) * np.sum(B ** 2))

    if denominateur == 0:
        return -1

    return np.sum(A * B) / denominateur

def extraire_patch(image, x, y, rayon):
    h, w = image.shape
    if x - rayon < 0 or y - rayon < 0:
        return None
    if x + rayon >= w or y + rayon >= h:
        return None

    return image[y-rayon:y+rayon+1, x-rayon:x+rayon+1]

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:
    raise FileNotFoundError("Vérifiez les chemins des images.")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

points1 = detecter_harris(image1, N)
points2 = detecter_harris(image2, N)

rayon = K // 2
correspondances = []

for x1, y1 in points1:
    patch1 = extraire_patch(gray1, x1, y1, rayon)

    if patch1 is None:
        continue

    meilleur_score = -1
    meilleur_point = None

    for x2, y2 in points2:
        patch2 = extraire_patch(gray2, x2, y2, rayon)

        if patch2 is None:
            continue

        score = calculer_zncc(patch1, patch2)

        if score > meilleur_score:
            meilleur_score = score
            meilleur_point = (x2, y2)

    if meilleur_point is not None:
        correspondances.append(
            ((x1, y1), meilleur_point, meilleur_score)
        )

print(f"Points Harris image 1 : {len(points1)}")
print(f"Points Harris image 2 : {len(points2)}")
print(f"Correspondances trouvées : {len(correspondances)}")

for i, (p1, p2, score) in enumerate(correspondances, 1):
    print(
        f"{i:02d}. Image 1 : {p1} -> Image 2 : {p2}"
        f" | ZNCC = {score:.4f}"
    )

resultat = cv2.drawMatches(
    image1,
    [cv2.KeyPoint(float(x), float(y), 5) for x, y in points1],
    image2,
    [cv2.KeyPoint(float(x), float(y), 5) for x, y in points2],
    [
        cv2.DMatch(i, points2.index(p2), 0)
        for i, (p1, p2, score) in enumerate(correspondances)
        if p1 in points1 and p2 in points2
    ],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB))
plt.title("Mise en correspondance des points de Harris par ZNCC")
plt.axis("off")
plt.show()
