# TP3 - IV-B.1
# Mise en correspondance entre une image objet (requête)
# et une image scène (base) par la méthode ZNCC.
#
# Pour chaque coin de l'image gauche :
# 1. Comparer sa fenêtre KxK avec chaque coin de l'image droite.
# 2. Conserver le point ayant la plus forte similarité.
# 3. Construire un tableau des correspondances :
#    coordonnées gauche, coordonnées droite, similarité.

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_REQUETE = "DataTP3/image.jpg"
IMAGE_SCENE = "DataTP3/tem0.jpg"

N = 50
K = 11
HARRIS_K = 0.04

def detecter_coins(image, n):
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

    y, x = np.where(maxima)
    coins = sorted(
        [(int(xi), int(yi), float(R[yi, xi])) for yi, xi in zip(y, x)],
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

def calculer_zncc(A, B):
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    A = A - np.mean(A)
    B = B - np.mean(B)

    denominateur = np.sqrt(np.sum(A**2) * np.sum(B**2))

    if denominateur == 0:
        return -1.0

    return float(np.sum(A * B) / denominateur)

# Chargement en niveaux de gris
requete = cv2.imread(IMAGE_REQUETE, cv2.IMREAD_GRAYSCALE)
scene = cv2.imread(IMAGE_SCENE, cv2.IMREAD_GRAYSCALE)

if requete is None or scene is None:
    raise FileNotFoundError("Vérifiez les chemins des images.")

# Détection des coins de Harris
coins_gauche = detecter_coins(requete, N)
coins_droite = detecter_coins(scene, N)

# Recherche du meilleur point correspondant
correspondances = []

for x1, y1 in coins_gauche:
    fenetre_gauche = extraire_fenetre(requete, x1, y1)

    if fenetre_gauche is None:
        continue

    meilleur_score = -1.0
    meilleur_point = None

    for x2, y2 in coins_droite:
        fenetre_droite = extraire_fenetre(scene, x2, y2)

        if fenetre_droite is None:
            continue

        score = calculer_zncc(fenetre_gauche, fenetre_droite)

        if score > meilleur_score:
            meilleur_score = score
            meilleur_point = (x2, y2)

    if meilleur_point is not None:
        correspondances.append(
            ((x1, y1), meilleur_point, meilleur_score)
        )

# Tableau des correspondances
print("\nTABLEAU DES CORRESPONDANCES")
print("-" * 75)
print(f"{'Point gauche':<20} {'Point droite':<20} {'ZNCC':>12}")
print("-" * 75)

for gauche, droite, score in correspondances:
    print(f"{str(gauche):<20} {str(droite):<20} {score:>12.4f}")

print("-" * 75)
print(f"Nombre de correspondances : {len(correspondances)}")

# Visualisation des correspondances
kp_gauche = [
    cv2.KeyPoint(float(g[0]), float(g[1]), 5)
    for g, d, s in correspondances
]

kp_droite = [
    cv2.KeyPoint(float(d[0]), float(d[1]), 5)
    for g, d, s in correspondances
]

matches = [
    cv2.DMatch(i, i, 0)
    for i in range(len(correspondances))
]

resultat = cv2.drawMatches(
    cv2.cvtColor(requete, cv2.COLOR_GRAY2BGR),
    kp_gauche,
    cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR),
    kp_droite,
    matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB))
plt.title("Correspondances entre l'objet et la scène - ZNCC")
plt.axis("off")
plt.show()
