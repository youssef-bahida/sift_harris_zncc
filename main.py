import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="TP3 - Vision par ordinateur", page_icon="🔎", layout="wide")

st.title("TP3 — Détection et mise en correspondance")
st.caption("SIFT • Harris • ZNCC | Analyse d'une image requête dans une ou plusieurs images scènes")

# ----------------------------- Utilitaires -----------------------------

def read_image(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de lire {uploaded_file.name}")
    return image

def gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def rgb(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def harris_response(gray, k=0.04, sigma=1.0):
    gray_f = gray.astype(np.float64)
    ix = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)

    m11 = cv2.GaussianBlur(ix * ix, (3, 3), sigma)
    m22 = cv2.GaussianBlur(iy * iy, (3, 3), sigma)
    m12 = cv2.GaussianBlur(ix * iy, (3, 3), sigma)

    trace = m11 + m22
    determinant = m11 * m22 - m12 * m12
    response = determinant - k * trace**2
    return ix, iy, trace, determinant, response

def strict_local_maxima(response):
    """Maxima stricts : réponse positive et supérieure aux 8 voisins."""
    maxima = np.zeros(response.shape, dtype=bool)
    if response.shape[0] < 3 or response.shape[1] < 3:
        return maxima

    center = response[1:-1, 1:-1]
    neighbors = np.stack([
        response[:-2, :-2], response[:-2, 1:-1], response[:-2, 2:],
        response[1:-1, :-2],                    response[1:-1, 2:],
        response[2:, :-2], response[2:, 1:-1], response[2:, 2:]
    ], axis=-1)

    maxima[1:-1, 1:-1] = (
        (center > 0) & np.all(center[..., None] > neighbors, axis=-1)
    )
    return maxima

def best_harris_points(gray, n=50, k=0.04, sigma=1.0):
    _, _, _, _, response = harris_response(gray, k, sigma)
    maxima = strict_local_maxima(response)
    ys, xs = np.where(maxima)
    candidates = [(int(x), int(y), float(response[y, x])) for y, x in zip(ys, xs)]
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:n], response, maxima

def gaussian_mask(size=16, sigma=8.0):
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2
    xx, yy = np.meshgrid(coords, coords)
    mask = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return mask / mask.max()

def compute_sift_patch(patch, sigma=8.0):
    """Descripteur pédagogique 4x4 cellules x 8 orientations = 128 valeurs."""
    patch = patch.astype(np.float64)
    if patch.shape != (16, 16):
        raise ValueError("Le patch doit mesurer 16 x 16 pixels.")

    ix = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(ix**2 + iy**2)
    orientation = (np.degrees(np.arctan2(iy, ix)) + 360) % 360
    weights = gaussian_mask(16, sigma)

    descriptor = np.zeros((4, 4, 8), dtype=np.float64)
    bins = np.floor(orientation / 45).astype(int) % 8

    for row in range(16):
        for col in range(16):
            cell_y, cell_x = row // 4, col // 4
            descriptor[cell_y, cell_x, bins[row, col]] += (
                magnitude[row, col] * weights[row, col]
            )

    vector = descriptor.ravel()
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector, magnitude, orientation, weights

def extract_patch(gray, x, y, window_size):
    radius = window_size // 2
    h, w = gray.shape
    if x - radius < 0 or y - radius < 0 or x + radius >= w or y + radius >= h:
        return None
    return gray[y-radius:y+radius+1, x-radius:x+radius+1]

def zncc(a, b):
    a = a.astype(np.float64) - np.mean(a)
    b = b.astype(np.float64) - np.mean(b)
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator <= 1e-12:
        return -1.0
    return float(np.sum(a * b) / denominator)

def zncc_matching(query_gray, scene_gray, n_points=50, window_size=11, harris_k=0.04):
    query_points, _, _ = best_harris_points(query_gray, n_points, harris_k)
    scene_points, _, _ = best_harris_points(scene_gray, n_points, harris_k)

    matches = []
    for x1, y1, _ in query_points:
        patch1 = extract_patch(query_gray, x1, y1, window_size)
        if patch1 is None:
            continue

        best_score = -1.0
        best_point = None
        for x2, y2, _ in scene_points:
            patch2 = extract_patch(scene_gray, x2, y2, window_size)
            if patch2 is None:
                continue
            score = zncc(patch1, patch2)
            if score > best_score:
                best_score = score
                best_point = (x2, y2)

        if best_point is not None:
            matches.append({
                "x gauche": x1, "y gauche": y1,
                "x droite": best_point[0], "y droite": best_point[1],
                "ZNCC": best_score
            })
    return matches, query_points, scene_points

def draw_harris_points(image, points):
    output = image.copy()
    for rank, (x, y, score) in enumerate(points, 1):
        cv2.circle(output, (x, y), 4, (0, 0, 255), 1)
        cv2.putText(output, str(rank), (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return output

def draw_zncc_matches(query, scene, matches, max_lines=50):
    left = query.copy()
    right = scene.copy()
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right

    # Les lignes sont dessinées avec une palette OpenCV déterministe.
    for i, item in enumerate(matches[:max_lines]):
        p1 = (int(item["x gauche"]), int(item["y gauche"]))
        p2 = (int(item["x droite"] + left.shape[1]), int(item["y droite"]))
        cv2.circle(canvas, p1, 4, (0, 0, 255), 1)
        cv2.circle(canvas, p2, 4, (0, 0, 255), 1)
        cv2.line(canvas, p1, p2, (0, 220, 0), 1)
    return canvas

# ----------------------------- Barre latérale -----------------------------

st.sidebar.header("Paramètres")
n_points = st.sidebar.slider("Nombre de points Harris (N)", 10, 200, 50, 10)
window_size = st.sidebar.select_slider("Taille de fenêtre ZNCC (K)", options=[5, 7, 9, 11, 13, 15, 17], value=11)
harris_k = st.sidebar.slider("Paramètre k de Harris", 0.02, 0.08, 0.04, 0.01)
gaussian_sigma = st.sidebar.slider("Sigma gaussien SIFT", 1.0, 12.0, 8.0, 0.5)
ratio_test = st.sidebar.slider("Seuil Lowe pour SIFT", 0.50, 0.95, 0.75, 0.05)

st.sidebar.markdown("---")
st.sidebar.info("Chargez une image objet (requête) et une ou plusieurs images scènes.")

query_file = st.file_uploader("Image objet / requête", type=["jpg", "jpeg", "png"])
scene_files = st.file_uploader(
    "Images scènes / base (une ou plusieurs)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if query_file is None:
    st.info("Commencez par importer l'image requête. Vous pourrez ensuite importer une ou plusieurs images scènes.")
    st.stop()

try:
    query_bgr = read_image(query_file)
    query_gray = gray_image(query_bgr)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if not scene_files:
    st.warning("Importez au moins une image scène pour activer les comparaisons.")
    st.image(rgb(query_bgr), caption="Image requête", use_container_width=True)

tab_sift, tab_harris, tab_zncc = st.tabs([
    "1. SIFT — Descripteur", "2. Harris — Détection", "3. ZNCC — Correspondances"
])

# ----------------------------- Onglet SIFT -----------------------------

with tab_sift:
    st.subheader("SIFT : gradients, masque gaussien et descripteurs")
    ix = cv2.Sobel(query_gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(query_gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(ix**2 + iy**2)
    orientation = (np.degrees(np.arctan2(iy, ix)) + 360) % 360
    mask = gaussian_mask(16, gaussian_sigma)

    c1, c2, c3 = st.columns(3)
    c1.image(rgb(query_bgr), caption="Image requête", use_container_width=True)
    c2.image(ix, caption="Gradient horizontal Ix", clamp=True, use_container_width=True)
    c3.image(iy, caption="Gradient vertical Iy", clamp=True, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.image(magnitude, caption="Magnitude Ig", clamp=True, use_container_width=True)
    c2.image(orientation, caption="Orientation Ior (degrés)", clamp=True, use_container_width=True)
    c3.image(mask, caption="Masque gaussien 16 x 16", clamp=True, use_container_width=True)

    st.markdown("**Descripteur pédagogique d'un patch 16 x 16**")
    if query_gray.shape[0] >= 16 and query_gray.shape[1] >= 16:
        max_y = query_gray.shape[0] - 16
        max_x = query_gray.shape[1] - 16
        colx, coly = st.columns(2)
        with colx:
            patch_x = st.slider("Position X du patch", 0, max_x, 0)
        with coly:
            patch_y = st.slider("Position Y du patch", 0, max_y, 0)
        patch = query_gray[patch_y:patch_y+16, patch_x:patch_x+16]
        descriptor, patch_mag, patch_ori, patch_mask = compute_sift_patch(patch, gaussian_sigma)
        p1, p2 = st.columns(2)
        p1.image(patch, caption="Patch 16 x 16", clamp=True, width=180)
        p2.line_chart(pd.DataFrame({"Valeur": descriptor}))
        st.caption(f"Dimension du descripteur : {descriptor.size} valeurs (4 x 4 cellules x 8 orientations).")
    else:
        st.warning("L'image doit mesurer au moins 16 x 16 pixels pour calculer le descripteur pédagogique.")

    if scene_files:
        st.markdown("**Détection et mise en correspondance SIFT (OpenCV)**")
        scene_choice = st.selectbox("Image scène pour SIFT", options=range(len(scene_files)),
                                    format_func=lambda i: scene_files[i].name)
        scene_bgr = read_image(scene_files[scene_choice])
        scene_gray = gray_image(scene_bgr)
        try:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(query_gray, None)
            kp2, des2 = sift.detectAndCompute(scene_gray, None)
            if des1 is None or des2 is None:
                st.warning("SIFT n'a pas trouvé suffisamment de descripteurs dans les deux images.")
            else:
                pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
                good = [m for m, n in pairs if m.distance < ratio_test * n.distance]
                visual = cv2.drawMatches(
                    query_bgr, kp1, scene_bgr, kp2, good[:100], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                st.write(f"Points SIFT requête : {len(kp1)} | Points SIFT scène : {len(kp2)} | Bonnes correspondances : {len(good)}")
                st.image(rgb(visual), caption="Correspondances SIFT (maximum 100 affichées)", use_container_width=True)
        except AttributeError:
            st.error("Cette installation OpenCV ne fournit pas SIFT. Utilisez opencv-python-headless récent.")

# ----------------------------- Onglet Harris -----------------------------

with tab_harris:
    st.subheader("Harris : réponse, maxima locaux et meilleurs points")
    ix_h, iy_h, trace, determinant, response = harris_response(query_gray, harris_k)
    points, _, maxima = best_harris_points(query_gray, n_points, harris_k)
    marked = draw_harris_points(query_bgr, points)

    c1, c2 = st.columns(2)
    c1.image(ix_h, caption="Gradient Ix", clamp=True, use_container_width=True)
    c2.image(iy_h, caption="Gradient Iy", clamp=True, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image(trace, caption="Trace de la matrice Harris", clamp=True, use_container_width=True)
    c2.image(determinant, caption="Déterminant de la matrice Harris", clamp=True, use_container_width=True)
    st.image(response, caption="Réponse de Harris R", clamp=True, use_container_width=True)
    st.image(rgb(marked), caption=f"{len(points)} meilleurs points de Harris (N demandé : {n_points})", use_container_width=True)
    st.metric("Maxima locaux strictement positifs", int(np.count_nonzero(maxima)))
    if points:
        st.dataframe(pd.DataFrame(
            [{"Rang": i, "x": x, "y": y, "Réponse Harris": score}
             for i, (x, y, score) in enumerate(points, 1)]
        ), use_container_width=True)
    else:
        st.warning("Aucun maximum local positif n'a été détecté.")

# ----------------------------- Onglet ZNCC -----------------------------

with tab_zncc:
    st.subheader("ZNCC : mise en correspondance avec une ou plusieurs scènes")
    st.caption("Chaque coin de la requête est associé indépendamment au coin de scène ayant le meilleur score ZNCC.")
    if not scene_files:
        st.info("Importez une ou plusieurs images scènes pour lancer la recherche.")
    else:
        all_summaries = []
        for idx, scene_file in enumerate(scene_files):
            with st.expander(f"Résultats pour {scene_file.name}", expanded=(idx == 0)):
                scene_bgr = read_image(scene_file)
                scene_gray = gray_image(scene_bgr)
                matches, query_points, scene_points = zncc_matching(
                    query_gray, scene_gray, n_points, window_size, harris_k
                )
                scores = [m["ZNCC"] for m in matches]
                mean_score = float(np.mean(scores)) if scores else float("nan")
                all_summaries.append({
                    "Image scène": scene_file.name,
                    "Coins requête": len(query_points),
                    "Coins scène": len(scene_points),
                    "Correspondances": len(matches),
                    "ZNCC moyen": mean_score
                })

                m1, m2, m3 = st.columns(3)
                m1.metric("Coins requête", len(query_points))
                m2.metric("Correspondances", len(matches))
                m3.metric("ZNCC moyen", f"{mean_score:.4f}" if np.isfinite(mean_score) else "N/A")

                if matches:
                    st.dataframe(pd.DataFrame(matches), use_container_width=True)
                    visual = draw_zncc_matches(query_bgr, scene_bgr, matches)
                    st.image(rgb(visual), caption=f"Correspondances ZNCC — {scene_file.name}", use_container_width=True)
                else:
                    st.warning("Aucune correspondance calculable. Vérifiez la taille des images et de la fenêtre K.")

        st.markdown("**Classement des images scènes**")
        summary_df = pd.DataFrame(all_summaries)
        summary_df = summary_df.sort_values("ZNCC moyen", ascending=False, na_position="last")
        st.dataframe(summary_df, use_container_width=True)
        st.caption("Le score moyen est indicatif : il ne prouve pas à lui seul que l'objet est présent dans l'image.")
