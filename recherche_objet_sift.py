# ============================================================
# TP3 : DETECTION ET MISE EN CORRESPONDANCE DE POINTS D'INTERET
#
# PARTIE II : RECHERCHE D'UNE PARTIE D'IMAGE (OBJET) DANS UNE IMAGE
#
# Question 1 :
# Ecrire un programme qui fait la recherche d'un objet ou d'une
# partie d'une image (requete) dans une image (scene), en se
# basant sur les fonctions predefinies de detection et matching.
#
# Methode utilisee : SIFT + BFMatcher + Ratio Test de Lowe
# ============================================================

import cv2
import matplotlib.pyplot as plt

# 1. Chargement des images
objet = cv2.imread("DataTP3/image.jpg")
scene = cv2.imread("DataTP3/tem0.jpg")

if objet is None or scene is None:
    raise FileNotFoundError("Images introuvables.")

# 2. Conversion en niveaux de gris
objet_gray = cv2.cvtColor(objet, cv2.COLOR_BGR2GRAY)
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

# 3. Creation du detecteur SIFT
sift = cv2.SIFT_create()

# 4. Detection des points d'interet et calcul des descripteurs
kp1, des1 = sift.detectAndCompute(objet_gray, None)
kp2, des2 = sift.detectAndCompute(scene_gray, None)

if des1 is None or des2 is None:
    raise ValueError("Aucun descripteur SIFT detecte.")

# 5. Mise en correspondance avec BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# 6. Filtrage des bonnes correspondances (Ratio Test de Lowe)
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 7. Affichage des correspondances
resultat = cv2.drawMatches(
    objet, kp1,
    scene, kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 8. Visualisation
plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB))
plt.title(f"Recherche d'objet par SIFT : {len(good_matches)} correspondances")
plt.axis("off")
plt.show()

# 9. Affichage des statistiques
print("Nombre de points dans l'objet :", len(kp1))
print("Nombre de points dans la scene :", len(kp2))
print("Nombre de correspondances trouvees :", len(good_matches))
