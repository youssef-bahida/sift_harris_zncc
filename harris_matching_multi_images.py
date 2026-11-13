# TP3 - Détection et mise en correspondance de points d'intérêt
# IV-B.2 : Mise en correspondance des points de Harris sur plusieurs images.
# L'image tem0.jpg sert de référence et les autres images sont comparées à elle.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_REFERENCE = "DataTP3/tem0.jpg"
DOSSIER_SCENES = Path("DataTP3")
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
    voisins = np.stack([
        R[:-2, :-2], R[:-2, 1:-1], R[:-2, 2:],
        R[1:-1, :-2], R[1:-1, 2:],
        R[2:, :-2], R[2:, 1:-1], R[2:, 2:]
    ], axis=-1)

    maxima = np.zeros(R.shape, dtype=bool)
    maxima[1:-1, 1:-1] = (
        (center > 0) & np.all(center[..., None] > voisins, axis=-1)
    )

    ys, xs = np.where(maxima)
    points = sorted(
        [(x, y, R[y, x]) for y, x in zip(ys, xs)],
        key=lambda p: p[2],
        reverse=True
    )

    return [(x, y) for x, y, _ in points[:n]]

def extraire_patch(image, x, y, rayon):
    h, w = image.shape
    if x - rayon < 0 or y - rayon < 0:
        return None
    if x + rayon >= w or y + rayon >= h:
        return None
    return image[y-rayon:y+rayon+1, x-rayon:x+rayon+1]

def zncc(patch1, patch2):
    A = patch1.astype(np.float64)
    B = patch2.astype(np.float64)

    A -= A.mean()
    B -= B.mean()

    denominateur = np.sqrt(np.sum(A**2) * np.sum(B**2))
    if denominateur == 0:
        return -1

    return np.sum(A * B) / denominateur

def comparer_images(reference, scene):
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    points_ref = detecter_harris(reference, N)
    points_scene = detecter_harris(scene, N)

    rayon = K // 2
    correspondances = []

    for x1, y1 in points_ref:
        patch1 = extraire_patch(gray_ref, x1, y1, rayon)
        if patch1 is None:
            continue

        meilleur_score = -1
        meilleur_point = None

        for x2, y2 in points_scene:
            patch2 = extraire_patch(gray_scene, x2, y2, rayon)
            if patch2 is None:
                continue

            score = zncc(patch1, patch2)

            if score > meilleur_score:
                meilleur_score = score
                meilleur_point = (x2, y2)

        if meilleur_point is not None:
            correspondances.append(((x1, y1), meilleur_point, meilleur_score))

    return correspondances

reference = cv2.imread(IMAGE_REFERENCE)

if reference is None:
    raise FileNotFoundError(f"Image de référence introuvable : {IMAGE_REFERENCE}")

fichiers = sorted(
    p for p in DOSSIER_SCENES.glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    and p.name != Path(IMAGE_REFERENCE).name
)

if not fichiers:
    raise FileNotFoundError("Aucune autre image de scène trouvée dans DataTP3.")

for fichier in fichiers:
    scene = cv2.imread(str(fichier))

    if scene is None:
        print(f"Image ignorée : {fichier}")
        continue

    correspondances = comparer_images(reference, scene)

    kp1 = [cv2.KeyPoint(float(p1[0]), float(p1[1]), 5)
           for p1, p2, score in correspondances]
    kp2 = [cv2.KeyPoint(float(p2[0]), float(p2[1]), 5)
           for p1, p2, score in correspondances]

    matches = [
        cv2.DMatch(i, i, 0)
        for i in range(len(correspondances))
    ]

    resultat = cv2.drawMatches(
        reference, kp1, scene, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    scores = [score for _, _, score in correspondances]
    moyenne = np.mean(scores) if scores else 0

    print(f"\nImage analysée : {fichier.name}")
    print(f"Correspondances : {len(correspondances)}")
    print(f"Score ZNCC moyen : {moyenne:.4f}")

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB))
    plt.title(f"tem0.jpg / {fichier.name} - ZNCC moyen : {moyenne:.3f}")
    plt.axis("off")
    plt.show()
