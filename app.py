"""Vista Tattica dall'Alto — Streamlit
Rilevamento giocatori (YOLO), validazione manuale squadra, posizionamento della palla,
calibrazione del campo tramite clic diretto sull'immagine, e proiezione 2D in scala reale.

Le operazioni pesanti (rilevamento YOLO, calcolo omografia + disegno della vista 2D)
sono messe in cache e vengono ricalcolate solo quando si preme il relativo tasto
"Applica", non ad ogni clic — per restare reattivi durante la validazione manuale.
"""
import csv
import io
import json
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates

from utils.detection import load_model, detect_players, assign_teams
from utils.pitch import LANDMARKS, draw_pitch, compute_homography, project_point

st.set_page_config(page_title="Vista Tattica dall'Alto", page_icon="⚽", layout="wide")

DISPLAY_MAX_W = 980
TEAM_COLORS = {0: (224, 102, 43), 1: (36, 97, 201), -1: (138, 138, 128)}  # RGB
TEAM_COLORS_HEX = {0: "#e0662b", 1: "#2461c9", -1: "#8a8a80"}
CALIB_COLOR = (255, 47, 208)
BALL_COLOR = (255, 255, 255)

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

MODE_PLAYERS, MODE_BALL, MODE_CALIB = "players", "ball", "calib"


# ----------------------------------------------------------------------
# Cache: modello YOLO (pesante da caricare) e rilevamento (pesante da calcolare)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_data(show_spinner="Rilevamento in corso (YOLOv8)...")
def run_detection_cached(frame_bytes: bytes, conf: float):
    """Cache per contenuto-immagine + soglia: se richiamata con gli stessi parametri
    non ricalcola, restituisce subito il risultato già calcolato."""
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    model = get_model()
    players = detect_players(model, frame, conf=conf)
    players = assign_teams(frame, players)
    return players


# ----------------------------------------------------------------------
# Stato
# ----------------------------------------------------------------------
def init_state():
    defaults = {
        "frame": None,            # np.array BGR, risoluzione originale
        "players": [],            # lista di dict id/bbox/x/y/team
        "ball": None,             # dict {"x":.., "y":..} oppure None
        "calib_points": [],       # lista di dict img/world/label
        "pending_click": None,    # ultimo clic in coordinate immagine originale (modo calib)
        "last_raw_click": None,   # per deduplicare i clic del componente
        "selected_player_id": None,
        "next_player_id": 0,
        "detected_preview": None,  # risultato cache in attesa di "Applica"
        "calib_dirty": True,       # True se la vista 2D va ricalcolata
        "view": None,              # ultima vista 2D applicata: {H, errors, rows, png}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def mark_dirty():
    st.session_state.calib_dirty = True


# ----------------------------------------------------------------------
# Utility di disegno
# ----------------------------------------------------------------------
def frame_to_display_image(frame_bgr, mode, players, ball, calib_points, pending_click, selected_id):
    """Converte il frame in una PIL.Image RGB ridimensionata per la UI, con le annotazioni
    disegnate sopra. Ritorna (img_pil, scale) dove scale converte coordinate
    immagine-mostrata -> coordinate immagine-originale (moltiplicare per 1/scale)."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, DISPLAY_MAX_W / w)
    disp_w, disp_h = int(w * scale), int(h * scale)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((disp_w, disp_h))
    draw = ImageDraw.Draw(img)

    # giocatori: sempre visibili per contesto, evidenziati in modo "players"
    for p in players:
        x, y = p["x"] * scale, p["y"] * scale
        color = TEAM_COLORS.get(p["team"], TEAM_COLORS[-1])
        r = 7 if mode == MODE_PLAYERS else 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0), width=2)
        if mode == MODE_PLAYERS and p["id"] == selected_id:
            draw.ellipse([x - r - 5, y - r - 5, x + r + 5, y + r + 5], outline=(255, 255, 0), width=3)

    # palla: sempre visibile per contesto
    if ball is not None:
        x, y = ball["x"] * scale, ball["y"] * scale
        r = 8 if mode == MODE_BALL else 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill=BALL_COLOR, outline=(0, 0, 0), width=2)
        if mode == MODE_BALL:
            draw.ellipse([x - r - 5, y - r - 5, x + r + 5, y + r + 5], outline=(255, 255, 0), width=3)

    if mode == MODE_CALIB:
        for i, c in enumerate(calib_points):
            x, y = c["img"][0] * scale, c["img"][1] * scale
            draw.line([x - 8, y, x + 8, y], fill=CALIB_COLOR, width=3)
            draw.line([x, y - 8, x, y + 8], fill=CALIB_COLOR, width=3)
            draw.text((x + 10, y - 10), str(i + 1), fill=CALIB_COLOR)
        if pending_click is not None:
            x, y = pending_click[0] * scale, pending_click[1] * scale
            draw.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 255, 255), width=3)

    return img, scale


def nearest_player(players, x, y, max_dist=28):
    best, best_d = None, max_dist
    for p in players:
        d = ((p["x"] - x) ** 2 + (p["y"] - y) ** 2) ** 0.5
        if d < best_d:
            best, best_d = p, d
    return best


def build_view(calib_points, players, ball):
    """Calcola omografia + disegna la vista 2D. Ritorna un dict con H, errors, rows, png (bytes)
    oppure None se la calibrazione non è ancora valida."""
    H, errors = compute_homography(calib_points)
    if H is None:
        return None

    fig, ax = plt.subplots(figsize=(11, 7.2))
    draw_pitch(ax)

    rows = []
    for p in players:
        px, py = project_point(H, p["x"], p["y"])
        color = TEAM_COLORS_HEX.get(p["team"], TEAM_COLORS_HEX[-1])
        ax.scatter(px, py, s=170, color=color, edgecolor="black", linewidth=1.1, zorder=5)
        ax.annotate(str(p["id"]), (px, py), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=7, color="white")
        rows.append({"tipo": "giocatore", "id": p["id"], "team": p["team"],
                     "x_img": round(p["x"], 1), "y_img": round(p["y"], 1),
                     "x_pitch_m": round(px, 2), "y_pitch_m": round(py, 2)})

    if ball is not None:
        bx, by = project_point(H, ball["x"], ball["y"])
        ax.scatter(bx, by, s=90, color="white", edgecolor="black", linewidth=1.3, zorder=6, marker="o")
        rows.append({"tipo": "palla", "id": "", "team": "",
                     "x_img": round(ball["x"], 1), "y_img": round(ball["y"], 1),
                     "x_pitch_m": round(bx, 2), "y_pitch_m": round(by, 2)})

    fig.patch.set_facecolor("#0e2e18")
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return {"H": H, "errors": errors, "rows": rows, "png": png_buf.getvalue()}


# ----------------------------------------------------------------------
# Sidebar — caricamento file e rilevamento
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚽ Vista Tattica 2D")
    st.caption("Carica un'immagine o un video: se è un video, viene usato il primo frame.")

    uploaded = st.file_uploader("Immagine o video", type=["png", "jpg", "jpeg", "webp", "bmp",
                                                            "mp4", "mov", "avi", "mkv", "webm", "m4v"])

    if uploaded is not None:
        suffix = "." + uploaded.name.split(".")[-1].lower()
        if st.session_state.get("_uploaded_name") != uploaded.name:
            st.session_state["_uploaded_name"] = uploaded.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            if suffix in VIDEO_EXT:
                cap = cv2.VideoCapture(tmp_path)
                ok, first_frame = cap.read()
                cap.release()
                if not ok or first_frame is None:
                    st.error("Non riesco a leggere il primo frame di questo video.")
                else:
                    st.session_state.frame = first_frame
                    st.session_state.players = []
                    st.session_state.ball = None
                    st.session_state.calib_points = []
                    st.session_state.detected_preview = None
                    st.session_state.view = None
                    st.session_state.calib_dirty = True
                    st.success(f"Primo frame estratto dal video ({first_frame.shape[1]}x{first_frame.shape[0]}).")
            elif suffix in IMAGE_EXT:
                frame = cv2.imread(tmp_path)
                if frame is None:
                    st.error("Non riesco a leggere questa immagine.")
                else:
                    st.session_state.frame = frame
                    st.session_state.players = []
                    st.session_state.ball = None
                    st.session_state.calib_points = []
                    st.session_state.detected_preview = None
                    st.session_state.view = None
                    st.session_state.calib_dirty = True
                    st.success(f"Immagine caricata ({frame.shape[1]}x{frame.shape[0]}).")

    st.divider()
    conf = st.slider("Soglia di confidenza rilevamento", 0.10, 0.80, 0.30, 0.05)

    if st.button("🔍 Rileva giocatori", type="primary", disabled=st.session_state.frame is None,
                 use_container_width=True):
        frame_bytes = cv2.imencode(".jpg", st.session_state.frame)[1].tobytes()
        st.session_state.detected_preview = run_detection_cached(frame_bytes, conf)

    if st.session_state.detected_preview is not None:
        prev = st.session_state.detected_preview
        n_a = sum(1 for p in prev if p["team"] == 0)
        n_b = sum(1 for p in prev if p["team"] == 1)
        n_x = sum(1 for p in prev if p["team"] == -1)
        st.info(f"Pronti **{len(prev)}** giocatori (A: {n_a} · B: {n_b} · non classificati: {n_x}).\n\n"
                "Non ancora applicati alla lista corrente.")
        c1, c2 = st.columns(2)
        if c1.button("✅ Applica alla lista", type="primary", use_container_width=True):
            st.session_state.players = [dict(p) for p in prev]
            st.session_state.next_player_id = (max((p["id"] for p in prev), default=-1)) + 1
            st.session_state.selected_player_id = None
            st.session_state.detected_preview = None
            mark_dirty()
            st.rerun()
        if c2.button("✕ Scarta", use_container_width=True):
            st.session_state.detected_preview = None
            st.rerun()

    st.divider()
    st.caption(f"Giocatori: **{len(st.session_state.players)}**  ·  "
               f"Palla: **{'posizionata' if st.session_state.ball else 'non posizionata'}**  ·  "
               f"Punti di calibrazione: **{len(st.session_state.calib_points)}**")


# ----------------------------------------------------------------------
# Corpo principale
# ----------------------------------------------------------------------
if st.session_state.frame is None:
    st.info("⬅️ Carica un'immagine o un video dalla barra laterale per iniziare.")
    st.stop()

mode_label = st.radio("Modalità", ["🧍 Giocatori", "⚽ Palla", "📐 Calibrazione campo"], horizontal=True)
mode = {"🧍 Giocatori": MODE_PLAYERS, "⚽ Palla": MODE_BALL, "📐 Calibrazione campo": MODE_CALIB}[mode_label]

col_img, col_panel = st.columns([2.2, 1])

with col_img:
    img_disp, scale = frame_to_display_image(
        st.session_state.frame, mode, st.session_state.players, st.session_state.ball,
        st.session_state.calib_points, st.session_state.pending_click, st.session_state.selected_player_id
    )
    click = streamlit_image_coordinates(img_disp, key=f"img_{mode}")

    if mode == MODE_PLAYERS:
        st.caption("Clicca su un giocatore per cambiarne la squadra o eliminarlo. "
                   "Clicca su un punto vuoto per aggiungerne uno mancante.")
    elif mode == MODE_BALL:
        st.caption("Clicca sulla posizione della palla per posizionarla (un nuovo clic la sposta).")
    else:
        st.caption("Clicca su un punto riconoscibile del campo, poi scegli a cosa corrisponde nel pannello a destra.")

# gestisci il clic (deduplicato rispetto all'ultimo elaborato)
if click is not None and click != st.session_state.last_raw_click:
    st.session_state.last_raw_click = click
    ox, oy = click["x"] / scale, click["y"] / scale  # -> coordinate immagine originale

    if mode == MODE_PLAYERS:
        hit = nearest_player(st.session_state.players, ox, oy)
        st.session_state.selected_player_id = hit["id"] if hit else "NEW:%f:%f" % (ox, oy)
    elif mode == MODE_BALL:
        st.session_state.ball = {"x": ox, "y": oy}
        mark_dirty()
    else:
        st.session_state.pending_click = (ox, oy)
    st.rerun()


# ----------------------------------------------------------------------
# Pannello laterale contestuale
# ----------------------------------------------------------------------
with col_panel:
    if mode == MODE_PLAYERS:
        sel = st.session_state.selected_player_id
        if sel is None:
            st.info("Nessun giocatore selezionato.")
        elif isinstance(sel, str) and sel.startswith("NEW:"):
            _, sx, sy = sel.split(":")
            st.warning("Nessun giocatore vicino a questo punto.")
            new_team = st.selectbox("Squadra del nuovo giocatore", ["Squadra A", "Squadra B", "Arbitro/altro"])
            if st.button("➕ Aggiungi qui", use_container_width=True):
                team_val = {"Squadra A": 0, "Squadra B": 1, "Arbitro/altro": -1}[new_team]
                st.session_state.players.append({
                    "id": st.session_state.next_player_id, "bbox": None, "conf": None,
                    "x": float(sx), "y": float(sy), "team": team_val,
                })
                st.session_state.next_player_id += 1
                st.session_state.selected_player_id = None
                mark_dirty()
                st.rerun()
        else:
            p = next((pl for pl in st.session_state.players if pl["id"] == sel), None)
            if p is None:
                st.session_state.selected_player_id = None
            else:
                team_name = {0: "Squadra A", 1: "Squadra B", -1: "Arbitro/altro"}[p["team"]]
                st.markdown(f"**Giocatore #{p['id']}** — squadra attuale: {team_name}")
                c1, c2 = st.columns(2)
                if c1.button("🟠 Squadra A", use_container_width=True):
                    p["team"] = 0; mark_dirty(); st.rerun()
                if c2.button("🔵 Squadra B", use_container_width=True):
                    p["team"] = 1; mark_dirty(); st.rerun()
                c3, c4 = st.columns(2)
                if c3.button("⚪ Arbitro/altro", use_container_width=True):
                    p["team"] = -1; mark_dirty(); st.rerun()
                if c4.button("🗑️ Elimina", use_container_width=True):
                    st.session_state.players = [pl for pl in st.session_state.players if pl["id"] != p["id"]]
                    st.session_state.selected_player_id = None
                    mark_dirty()
                    st.rerun()

        st.divider()
        with st.expander("Reimposta tutti i giocatori"):
            if st.button("Svuota la lista giocatori", use_container_width=True):
                st.session_state.players = []
                st.session_state.selected_player_id = None
                mark_dirty()
                st.rerun()

    elif mode == MODE_BALL:
        if st.session_state.ball is None:
            st.info("Palla non ancora posizionata. Clicca sull'immagine a sinistra.")
        else:
            b = st.session_state.ball
            st.markdown(f"**Palla** — posizione immagine: ({b['x']:.0f}, {b['y']:.0f})")
            if st.button("🗑️ Rimuovi palla", use_container_width=True):
                st.session_state.ball = None
                mark_dirty()
                st.rerun()

    else:  # calib mode
        if st.session_state.pending_click is None:
            st.info("Clicca un punto sull'immagine a sinistra.")
        else:
            landmark = st.selectbox("A cosa corrisponde questo punto?", list(LANDMARKS.keys()))
            if st.button("✅ Conferma punto", type="primary", use_container_width=True):
                st.session_state.calib_points.append({
                    "img": st.session_state.pending_click, "world": LANDMARKS[landmark], "label": landmark,
                })
                st.session_state.pending_click = None
                mark_dirty()
                st.rerun()
            if st.button("Annulla", use_container_width=True):
                st.session_state.pending_click = None
                st.rerun()

        if st.session_state.calib_points:
            st.divider()
            st.markdown("**Punti di calibrazione:**")
            for i, c in enumerate(st.session_state.calib_points):
                cc1, cc2 = st.columns([5, 1])
                cc1.caption(f"{i+1}. {c['label']}")
                if cc2.button("✕", key=f"del_calib_{i}"):
                    st.session_state.calib_points.pop(i)
                    mark_dirty()
                    st.rerun()


# ----------------------------------------------------------------------
# Vista 2D + esportazione
# ----------------------------------------------------------------------
st.divider()
st.header("Vista Tattica dall'Alto — proiezione 2D")

n_calib = len(st.session_state.calib_points)
if n_calib < 4:
    st.warning(f"Servono almeno 4 punti di calibrazione non allineati (attuali: {n_calib}).")
else:
    if st.session_state.calib_dirty:
        st.warning("⚠️ Punti, giocatori o palla modificati dall'ultimo aggiornamento della vista 2D.")
    apply_label = "🔄 Applica calibrazione / aggiorna vista 2D" if st.session_state.view is not None \
        else "📐 Genera vista 2D"
    if st.button(apply_label, type="primary"):
        result = build_view(st.session_state.calib_points, st.session_state.players, st.session_state.ball)
        if result is None:
            st.error("I punti scelti sono troppo allineati o coincidenti: la calibrazione non converge.")
        else:
            st.session_state.view = result
            st.session_state.calib_dirty = False
            st.rerun()

    view = st.session_state.view
    if view is not None:
        with st.expander("Qualità della calibrazione (errore di riproiezione)"):
            for c, e in zip(st.session_state.calib_points, view["errors"]):
                st.caption(f"{c['label']}: {e:.3f} m")
            st.caption(f"**Errore medio: {view['errors'].mean():.3f} m · "
                       f"massimo: {view['errors'].max():.3f} m**")
            if view["errors"].max() > 2.0:
                st.warning("Errore massimo elevato: ricontrolla i punti più imprecisi.")

        st.image(view["png"], use_container_width=True)

        st.subheader("Esportazione")
        rows = view["rows"]
        csv_buf = io.StringIO()
        fieldnames = list(rows[0].keys()) if rows else \
            ["tipo", "id", "team", "x_img", "y_img", "x_pitch_m", "y_pitch_m"]
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇️ CSV", csv_buf.getvalue(), "giocatori_2d.csv", "text/csv",
                            use_container_width=True)
        e2.download_button("⬇️ JSON", json.dumps(rows, indent=2), "giocatori_2d.json",
                            "application/json", use_container_width=True)
        e3.download_button("⬇️ Immagine PNG", view["png"], "vista_tattica_2d.png",
                            "image/png", use_container_width=True)
    else:
        st.info("Premi il pulsante qui sopra per generare la vista 2D.")
