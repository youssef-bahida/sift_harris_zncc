# TP3 - IV-B.2
# Recherche d'un objet dans une base contenant plusieurs images.
# Pour chaque coin de l'image requête, rechercher le meilleur
# point correspondant dans chaque image de la base avec ZNCC.
#
# Tableau des résultats :
# coordonnées image gauche ; coordonnées image droite ; similarité.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_REQUETE = "DataTP3/image.jpg"
DOSSIER_BASE = Path("DataTP3")

N = 50
K = 11
HARRIS_K = 0.04

def detecter_coins(image, n):
    image = image.astype(np.float64)

    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    M11 = cv2.GaussianBlur(Ix * Ix, (3, 3), 1)
    M22 = cv2.GaussianBlur(Iy * Iy, (3, 3), 1)
    M12 = cv2.GaussianBlur(Ix * Iy, (3, 3), 1)

    trace = M11 + M22
    det = M11 * M22 - M12 ** 2
    R = det - HARRIS_K * trace ** 2

    centre = R[1:-1, 1:-1]

    voisins = np.stack([
        R[:-2, :-2], R[:-2, 1:-1], R[:-2, 2:],
        R[1:-1, :-2], R[1:-1, 2:],
        R[2:, :-2], R[2:, 1:-1], R[2:, 2:]
    ], axis=-1)

    maxima = np.zeros(R.shape, dtype=bool)
    maxima[1:-1, 1:-1] = (
        (centre > 0) & np.all(centre[..., None] > voisins, axis=-1)
    )

    ys, xs = np.where(maxima)

    coins = sorted(
        [(int(x), int(y), float(R[y, x])) for y, x in zip(ys, xs)],
        key=lambda p: p[2],
        reverse=True
    )

    return [(x, y) for x, y, score in coins[:n]]

def extraire_fenetre(image, x, y):
    rayon = K // 2

    if (x - rayon < 0 or y - rayon < 0 or
        x + rayon >= image.shape[1] or
        y + rayon >= image.shape[0]):
        return None

    return image[y-rayon:y+rayon+1, x-rayon:x+rayon+1]

def zncc(A, B):
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    A -= A.mean()
    B -= B.mean()

    denominateur = np.sqrt(np.sum(A**2) * np.sum(B**2))

    if denominateur == 0:
        return -1.0

    return float(np.sum(A * B) / denominateur)

def rechercher(requete, scene, coins_requete):
    coins_scene = detecter_coins(scene, N)
    correspondances = []

    for x1, y1 in coins_requete:
        patch1 = extraire_fenetre(requete, x1, y1)

        if patch1 is None:
            continue

        meilleur_score = -1.0
        meilleur_point = None

        for x2, y2 in coins_scene:
            patch2 = extraire_fenetre(scene, x2, y2)

            if patch2 is None:
                continue

            score = zncc(patch1, patch2)

            if score > meilleur_score:
                meilleur_score = score
                meilleur_point = (x2, y2)

        if meilleur_point is not None:
            correspondances.append(
                ((x1, y1), meilleur_point, meilleur_score)
            )

    return correspondances

# Chargement de l'image requête
requete = cv2.imread(IMAGE_REQUETE, cv2.IMREAD_GRAYSCALE)

if requete is None:
    raise FileNotFoundError(f"Image requête introuvable : {IMAGE_REQUETE}")

coins_requete = detecter_coins(requete, N)

# Recherche de toutes les images de la base
fichiers = sorted(
    p for p in DOSSIER_BASE.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    and p.name != Path(IMAGE_REQUETE).name
)

if not fichiers:
    raise FileNotFoundError("Aucune image trouvée dans la base.")

resultats = []

for fichier in fichiers:
    scene = cv2.imread(str(fichier), cv2.IMREAD_GRAYSCALE)

    if scene is None:
        continue

    correspondances = rechercher(requete, scene, coins_requete)

    scores = [score for gauche, droite, score in correspondances]
    moyenne = float(np.mean(scores)) if scores else -1.0

    resultats.append((fichier, scene, correspondances, moyenne))

# Classement des images de la base par score moyen décroissant
resultats.sort(key=lambda r: r[3], reverse=True)

for fichier, scene, correspondances, moyenne in resultats:
    print(f"\nIMAGE DE LA BASE : {fichier.name}")
    print("-" * 75)
    print(f"{'Point gauche':<20} {'Point droite':<20} {'ZNCC':>12}")
    print("-" * 75)

    for gauche, droite, score in correspondances:
        print(f"{str(gauche):<20} {str(droite):<20} {score:>12.4f}")

    print("-" * 75)
    print(f"Nombre de correspondances : {len(correspondances)}")
    print(f"Similarité moyenne : {moyenne:.4f}")

    kp1 = [
        cv2.KeyPoint(float(g[0]), float(g[1]), 5)
        for g, d, s in correspondances
    ]

    kp2 = [
        cv2.KeyPoint(float(d[0]), float(d[1]), 5)
        for g, d, s in correspondances
    ]

    matches = [
        cv2.DMatch(i, i, 0)
        for i in range(len(correspondances))
    ]

    resultat = cv2.drawMatches(
        cv2.cvtColor(requete, cv2.COLOR_GRAY2BGR),
        kp1,
        cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR),
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB))
    plt.title(f"{fichier.name} - Similarité moyenne ZNCC : {moyenne:.3f}")
    plt.axis("off")
    plt.show()

print("\nCLASSEMENT FINAL DE LA BASE")
for rang, (fichier, scene, correspondances, moyenne) in enumerate(resultats, 1):
    print(f"{rang}. {fichier.name} : ZNCC moyen = {moyenne:.4f}")
