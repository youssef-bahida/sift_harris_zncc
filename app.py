import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ==========================================================
# TP3 - VISION PAR ORDINATEUR
# SIFT | HARRIS | ZNCC
# ==========================================================

st.set_page_config(
    page_title="TP3 | Computer Vision",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- DESIGN -----------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1450px;
    }
    h1 {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }
    h2 {
        font-size: 1.25rem !important;
    }
    h3 {
        font-size: 1rem !important;
    }
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.07);
        border: 1px solid rgba(128,128,128,0.15);
        padding: 10px 14px;
        border-radius: 9px;
    }
    [data-testid="stImage"] img {
        border-radius: 7px;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
    }
    .subtitle {
        color: #8993a4;
        font-size: 0.85rem;
        margin-top: -12px;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #8993a4;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- DEMONSTRATION ----------------------

def create_demo():
    rng = np.random.default_rng(42)

    obj = np.full((220, 220, 3), 235, np.uint8)
    cv2.rectangle(obj, (25, 25), (195, 195), (40, 100, 180), -1)
    cv2.rectangle(obj, (45, 45), (175, 175), (240, 210, 70), -1)
    cv2.circle(obj, (110, 110), 43, (30, 30, 210), -1)
    cv2.circle(obj, (110, 110), 17, (255, 255, 255), -1)
    cv2.line(obj, (45, 165), (175, 55), (0, 0, 0), 5)
    cv2.putText(obj, "TP3", (72, 190),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

    scene1 = np.full((420, 620, 3), 225, np.uint8)
    cv2.rectangle(scene1, (15, 15), (605, 405), (190, 190, 190), 3)
    resized = cv2.resize(obj, (190, 190))
    scene1[110:300, 210:400] = resized
    cv2.rectangle(scene1, (50, 80), (150, 170), (80, 150, 80), -1)
    cv2.circle(scene1, (510, 300), 50, (180, 90, 30), -1)

    scene2 = np.full((420, 620, 3), 230, np.uint8)
    cv2.rectangle(scene2, (15, 15), (605, 405), (190, 190, 190), 3)

    for _ in range(12):
        x = int(rng.integers(40, 570))
        y = int(rng.integers(40, 370))
        color = tuple(int(v) for v in rng.integers(30, 220, 3))
        cv2.circle(scene2, (x, y), 20, color, -1)

    cv2.rectangle(scene2, (80, 100), (220, 260), (100, 160, 80), -1)
    cv2.rectangle(scene2, (350, 90), (520, 270), (160, 100, 170), -1)
    cv2.line(scene2, (40, 340), (570, 340), (0, 0, 0), 4)

    return obj, scene1, scene2


def load_demo():
    paths = [
        Path("DataTP3/image.jpg"),
        Path("DataTP3/tem0.jpg"),
        Path("DataTP3/tem1.jpg")
    ]

    if all(p.exists() for p in paths):
        images = [cv2.imread(str(p)) for p in paths]
        if all(im is not None for im in images):
            return images

    return list(create_demo())


# ---------------------- UTILITAIRES ------------------------

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rgb(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_upload(file):
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image invalide")
    return img


# ---------------------- HARRIS -----------------------------

def harris(gray_img, k=0.04, sigma=1):
    im = gray_img.astype(np.float64)

    ix = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

    a = cv2.GaussianBlur(ix * ix, (3, 3), sigma)
    b = cv2.GaussianBlur(iy * iy, (3, 3), sigma)
    c = cv2.GaussianBlur(ix * iy, (3, 3), sigma)

    trace = a + b
    det = a * b - c * c
    response = det - k * trace**2

    return ix, iy, trace, det, response


def harris_points(im, n=50, k=0.04):
    _, _, _, _, response = harris(im, k)
    maxima = np.zeros(response.shape, dtype=bool)

    if min(response.shape) >= 3:
        center = response[1:-1, 1:-1]
        neighbors = np.stack([
            response[:-2, :-2], response[:-2, 1:-1],
            response[:-2, 2:], response[1:-1, :-2],
            response[1:-1, 2:], response[2:, :-2],
            response[2:, 1:-1], response[2:, 2:]
        ], axis=-1)

        maxima[1:-1, 1:-1] = (
            (center > 0) &
            np.all(center[..., None] > neighbors, axis=-1)
        )

    ys, xs = np.where(maxima)
    points = [(int(x), int(y), float(response[y, x]))
              for y, x in zip(ys, xs)]

    points.sort(key=lambda p: p[2], reverse=True)
    return points[:n], response, maxima


def draw_points(img, points):
    out = img.copy()
    for i, (x, y, _) in enumerate(points, 1):
        cv2.circle(out, (x, y), 4, (0, 0, 255), 1)
    return out


# ---------------------- SIFT PEDAGOGIQUE -------------------

def gaussian_mask(size=16, sigma=8):
    coords = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(coords, coords)
    mask = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return mask / mask.max()


def sift_descriptor(patch, sigma=8):
    patch = patch.astype(float)

    ix = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.sqrt(ix**2 + iy**2)
    ori = (np.degrees(np.arctan2(iy, ix)) + 360) % 360
    mask = gaussian_mask(16, sigma)

    desc = np.zeros((4, 4, 8))
    bins = (ori // 45).astype(int) % 8

    for y in range(16):
        for x in range(16):
            desc[y//4, x//4, bins[y, x]] += mag[y, x] * mask[y, x]

    vector = desc.ravel()
    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    return vector, mag, ori


def sift_match(query, scene, ratio):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray(query), None)
    kp2, des2 = sift.detectAndCompute(gray(scene), None)

    if des1 is None or des2 is None:
        return None, len(kp1), len(kp2), 0

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)

    good = [
        p[0] for p in pairs
        if len(p) == 2 and p[0].distance < ratio * p[1].distance
    ]

    visual = cv2.drawMatches(
        query, kp1, scene, kp2, good[:80], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return visual, len(kp1), len(kp2), len(good)


# ---------------------- ZNCC -------------------------------

def zncc(a, b):
    a = a.astype(float) - np.mean(a)
    b = b.astype(float) - np.mean(b)
    den = np.sqrt(np.sum(a*a) * np.sum(b*b))
    return float(np.sum(a*b) / den) if den > 1e-12 else -1


def patch_at(im, x, y, k):
    r = k // 2
    h, w = im.shape
    if x-r < 0 or y-r < 0 or x+r >= w or y+r >= h:
        return None
    return im[y-r:y+r+1, x-r:x+r+1]


def zncc_match(query, scene, n, k, hk):
    q, s = gray(query), gray(scene)
    qp, _, _ = harris_points(q, n, hk)
    sp, _, _ = harris_points(s, n, hk)

    matches = []

    for x1, y1, _ in qp:
        p1 = patch_at(q, x1, y1, k)
        if p1 is None:
            continue

        best = -1
        best_xy = None

        for x2, y2, _ in sp:
            p2 = patch_at(s, x2, y2, k)
            if p2 is None:
                continue

            score = zncc(p1, p2)
            if score > best:
                best, best_xy = score, (x2, y2)

        if best_xy is not None:
            matches.append({
                "X requête": x1,
                "Y requête": y1,
                "X scène": best_xy[0],
                "Y scène": best_xy[1],
                "ZNCC": best
            })

    return matches, qp, sp


def draw_zncc(query, scene, matches):
    h = max(query.shape[0], scene.shape[0])
    w1 = query.shape[1]
    canvas = np.zeros((h, w1 + scene.shape[1], 3), np.uint8)

    canvas[:query.shape[0], :w1] = query
    canvas[:scene.shape[0], w1:] = scene

    for m in matches[:60]:
        p1 = (m["X requête"], m["Y requête"])
        p2 = (m["X scène"] + w1, m["Y scène"])
        cv2.line(canvas, p1, p2, (0, 220, 0), 1)
        cv2.circle(canvas, p1, 3, (0, 0, 255), 1)
        cv2.circle(canvas, p2, 3, (0, 0, 255), 1)

    return canvas


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🔬 TP3 | Paramètres")

mode = st.sidebar.radio(
    "Source des images",
    ["Démonstration", "Mes images"],
    horizontal=False
)

st.sidebar.divider()

n_points = st.sidebar.slider("Points Harris (N)", 10, 200, 50, 10)
k_window = st.sidebar.select_slider(
    "Fenêtre ZNCC (K)",
    options=[5, 7, 9, 11, 13, 15, 17],
    value=11
)
harris_k = st.sidebar.slider("Coefficient Harris", 0.02, 0.08, 0.04, 0.01)
sigma = st.sidebar.slider("Sigma gaussien", 1.0, 12.0, 8.0, 0.5)
ratio = st.sidebar.slider("Ratio Lowe", 0.50, 0.95, 0.75, 0.05)

st.sidebar.divider()
st.sidebar.caption("TP3 — Vision par ordinateur")


# ==========================================================
# SOURCE DES IMAGES
# ==========================================================

if mode == "Démonstration":
    demo = load_demo()
    query, scenes = demo[0], demo[1:]
    scene_names = ["tem0 — Scène 1", "tem1 — Scène 2"]

else:
    q_file = st.sidebar.file_uploader(
        "Image requête", type=["jpg", "jpeg", "png"]
    )
    s_files = st.sidebar.file_uploader(
        "Images scènes",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if q_file is None or not s_files:
        st.title("TP3 — Vision par ordinateur")
        st.info("Importe une image requête et au moins une image scène.")
        st.stop()

    query = read_upload(q_file)
    scenes = [read_upload(f) for f in s_files]
    scene_names = [f.name for f in s_files]


# ==========================================================
# HEADER + NAVIGATION
# ==========================================================

st.title("TP3 — Vision par ordinateur")
st.markdown(
    '<p class="subtitle">Détection de points d’intérêt et mise en correspondance</p>',
    unsafe_allow_html=True
)

nav = st.radio(
    "Méthode",
    ["Vue générale", "SIFT", "Harris", "ZNCC"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# ==========================================================
# VUE GENERALE
# ==========================================================

if nav == "Vue générale":

    st.subheader("Espace de travail")

    c1, c2, c3 = st.columns(3)

    c1.metric("Image requête", "1")
    c2.metric("Images scènes", len(scenes))
    c3.metric("Méthodes", "3")

    st.markdown("#### Images chargées")

    cols = st.columns(min(3, len(scenes) + 1))

    with cols[0]:
        st.image(rgb(query), caption="Image requête",
                 use_container_width=True)

    for i, scene in enumerate(scenes):
        with cols[(i + 1) % len(cols)]:
            st.image(rgb(scene), caption=scene_names[i],
                     use_container_width=True)

    st.divider()

    st.markdown("#### Méthodes disponibles")

    a, b, c = st.columns(3)

    with a:
        st.markdown("**01 — SIFT**")
        st.caption("Extraction de descripteurs locaux et correspondances.")

    with b:
        st.markdown("**02 — Harris**")
        st.caption("Détection et classement des points d'intérêt.")

    with c:
        st.markdown("**03 — ZNCC**")
        st.caption("Comparaison des fenêtres locales et appariement.")

    st.info("Sélectionne une méthode dans la barre supérieure pour commencer.")


# ==========================================================
# SIFT
# ==========================================================

elif nav == "SIFT":

    st.subheader("SIFT — Scale-Invariant Feature Transform")

    im = gray(query).astype(float)

    ix = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(ix**2 + iy**2)
    ori = (np.degrees(np.arctan2(iy, ix)) + 360) % 360

    st.markdown("**Analyse des gradients**")

    c1, c2, c3 = st.columns(3)

    c1.image(ix, caption="Gradient Ix", clamp=True,
             use_container_width=True)
    c2.image(iy, caption="Gradient Iy", clamp=True,
             use_container_width=True)
    c3.image(mag, caption="Magnitude Ig", clamp=True,
             use_container_width=True)

    with st.expander("Orientation et masque gaussien"):
        c1, c2 = st.columns(2)
        c1.image(ori, caption="Orientation Ior", clamp=True,
                 use_container_width=True)
        c2.image(gaussian_mask(16, sigma),
                 caption="Masque gaussien 16 × 16",
                 clamp=True, use_container_width=True)

    st.markdown("**Descripteur pédagogique — 128 dimensions**")

    if min(im.shape) >= 16:
        max_x = im.shape[1] - 16
        max_y = im.shape[0] - 16

        c1, c2 = st.columns(2)
        x = c1.slider("Position X", 0, max_x, 0)
        y = c2.slider("Position Y", 0, max_y, 0)

        patch = im[y:y+16, x:x+16]
        desc, _, _ = sift_descriptor(patch, sigma)

        c1, c2 = st.columns([1, 2])
        c1.image(patch, caption="Patch 16 × 16", clamp=True, width=180)
        c2.line_chart(pd.DataFrame({"Descripteur": desc}))
        st.caption("4 × 4 cellules × 8 orientations = 128 valeurs.")

    st.divider()
    st.markdown("**Correspondances SIFT**")

    idx = st.selectbox(
        "Image scène",
        range(len(scenes)),
        format_func=lambda i: scene_names[i]
    )

    visual, n1, n2, good = sift_match(query, scenes[idx], ratio)

    a, b, c = st.columns(3)
    a.metric("Points requête", n1)
    b.metric("Points scène", n2)
    c.metric("Bonnes correspondances", good)

    if visual is not None:
        st.image(rgb(visual), caption="Matching SIFT",
                 use_container_width=True)
    else:
        st.warning("Aucun descripteur exploitable.")


# ==========================================================
# HARRIS
# ==========================================================

elif nav == "Harris":

    st.subheader("Harris — Détection des points d'intérêt")

    im = gray(query)
    ix, iy, trace, det, response = harris(im, harris_k)
    points, response, maxima = harris_points(im, n_points, harris_k)

    a, b, c = st.columns(3)
    a.metric("Maxima locaux", int(np.count_nonzero(maxima)))
    b.metric("Points sélectionnés", len(points))
    c.metric("Paramètre k", harris_k)

    st.markdown("**Réponse de Harris**")

    c1, c2 = st.columns(2)

    c1.image(response, caption="Carte de réponse R",
             clamp=True, use_container_width=True)
    c2.image(rgb(draw_points(query, points)),
             caption="Meilleurs points détectés",
             use_container_width=True)

    with st.expander("Détails des calculs intermédiaires"):
        c1, c2 = st.columns(2)
        c1.image(ix, caption="Gradient Ix", clamp=True,
                 use_container_width=True)
        c2.image(iy, caption="Gradient Iy", clamp=True,
                 use_container_width=True)

        c1, c2 = st.columns(2)
        c1.image(trace, caption="Trace", clamp=True,
                 use_container_width=True)
        c2.image(det, caption="Déterminant", clamp=True,
                 use_container_width=True)

    with st.expander("Tableau des meilleurs points"):
        df = pd.DataFrame([
            {"Rang": i, "X": x, "Y": y, "Réponse": score}
            for i, (x, y, score) in enumerate(points, 1)
        ])
        st.dataframe(df, use_container_width=True, height=250)


# ==========================================================
# ZNCC
# ==========================================================

elif nav == "ZNCC":

    st.subheader("ZNCC — Mise en correspondance")

    st.caption(
        "Comparaison des fenêtres locales autour des points Harris."
    )

    summaries = []

    for i, scene in enumerate(scenes):

        matches, qp, sp = zncc_match(
            query, scene, n_points, k_window, harris_k
        )

        scores = [m["ZNCC"] for m in matches]
        mean_score = float(np.mean(scores)) if scores else float("nan")

        summaries.append({
            "Scène": scene_names[i],
            "Points requête": len(qp),
            "Correspondances": len(matches),
            "ZNCC moyen": mean_score
        })

        with st.expander(
            f"{scene_names[i]} — {len(matches)} correspondances",
            expanded=(i == 0)
        ):

            a, b, c = st.columns(3)
            a.metric("Points requête", len(qp))
            b.metric("Correspondances", len(matches))
            c.metric(
                "ZNCC moyen",
                f"{mean_score:.4f}" if np.isfinite(mean_score) else "N/A"
            )

            if matches:
                st.image(
                    rgb(draw_zncc(query, scene, matches)),
                    caption="Mise en correspondance",
                    use_container_width=True
                )

                with st.expander("Afficher le tableau des correspondances"):
                    st.dataframe(
                        pd.DataFrame(matches),
                        use_container_width=True,
                        height=220
                    )
            else:
                st.warning("Aucune correspondance calculable.")

    st.divider()
    st.markdown("**Comparaison des scènes**")

    df = pd.DataFrame(summaries).sort_values(
        "ZNCC moyen", ascending=False, na_position="last"
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "Le score moyen ZNCC est indicatif. "
        "Il ne constitue pas à lui seul une preuve de présence de l'objet."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "TP3 | Vision par ordinateur | SIFT • Harris • ZNCC"
)
