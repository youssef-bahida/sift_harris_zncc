# SIFT, Harris et ZNCC – Détection et reconnaissance de points d'intérêt

## Application en ligne

🚀 **[Accéder à l'application Streamlit](https://yb-sift-harris-zncc.streamlit.app/)**

## Présentation du projet

Ce projet consiste à étudier et à implémenter différentes méthodes de détection, de description et de mise en correspondance de points d'intérêt dans les images.

L'application interactive permet de visualiser et de comparer les résultats des méthodes SIFT, Harris et ZNCC à travers des exemples prédéfinis ou des images importées par l'utilisateur.

## Méthodes étudiées

### 1. SIFT (Scale-Invariant Feature Transform)
- Calcul des gradients.
- Calcul de la magnitude et de l'orientation.
- Construction du masque gaussien.
- Calcul du descripteur SIFT de 128 dimensions.
- Détection et mise en correspondance des caractéristiques.

### 2. Harris Corner Detector
- Calcul des gradients de l'image.
- Calcul des dérivées secondes.
- Calcul de la réponse de Harris.
- Détection des maxima locaux.
- Sélection des meilleurs points d'intérêt.

### 3. ZNCC (Zero-Mean Normalized Cross-Correlation)
- Comparaison des fenêtres autour des points d'intérêt.
- Calcul du coefficient de corrélation ZNCC.
- Recherche de correspondances entre une image requête et plusieurs images scènes.
- Classement des scènes selon leur similarité.

## Fonctionnalités de l'application

- Exemples intégrés pour tester les méthodes sans importer d'images.
- Possibilité de téléverser ses propres images.
- Visualisation des points d'intérêt détectés.
- Affichage des correspondances entre les images.
- Paramétrage des principaux paramètres des algorithmes.
- Comparaison des résultats sur plusieurs images scènes.

## Technologies utilisées

- Python
- OpenCV
- NumPy
- Pandas
- Streamlit
- Matplotlib

## Installation locale

Cloner le dépôt :

```bash
git clone https://github.com/youssef-bahida/sift_harris_znxcc.git
cd sift_harris_znxcc
```

Créer et activer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'application :

```bash
streamlit run app.py
```

## Structure du projet

```text
sift_harris_znxcc/
│
├── app.py
├── requirements.txt
├── .gitignore
├── DataTP3/
│   ├── image.jpg
│   ├── tem0.jpg
│   └── tem1.jpg
│
├── calcul_descripteur_sift.py
├── calcul_gradient_sift.py
├── echantillonnage_sift.py
├── masque_gaussien_sift.py
├── recherche_objet_sift.py
│
├── harris_gradients.py
├── harris_response.py
├── harris_second_derivatives.py
├── harris_local_maxima.py
├── harris_meilleurs_points.py
├── harris_matching_multi_images.py
├── harris_matching_zncc.py
│
└── matching_objet_scene_zncc.py
```

## Auteur

**Youssef Bahida**

- GitHub : https://github.com/youssef-bahida
- Repository : https://github.com/youssef-bahida/sift_harris_znxcc
