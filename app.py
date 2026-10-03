"""Vista Tattica dall'Alto — Streamlit
Rilevamento giocatori (YOLO), validazione manuale squadra, posizionamento della palla,
calibrazione del campo tramite clic diretto sull'immagine, e proiezione 2D in scala reale.

Tutte le operazioni pesanti o che richiedono più clic in sequenza sono pensate per
restare reattive:
- il modello YOLO e il rilevamento sono in cache (si ricalcolano solo se cambi
  immagine o soglia);
- le correzioni manuali ai giocatori (squadra, eliminazioni, nuovi giocatori) e lo
  spostamento della palla restano "in bozza" mentre clicchi, e diventano definitive
  solo quando premi "✅ Applica tutte le modifiche" — così puoi correggere più
  giocatori sbagliati di fila senza attese tra un clic e l'altro;
- il calcolo dell'omografia e il disegno della vista 2D si aggiornano solo premendo
  "Applica calibrazione / aggiorna vista 2D".
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
from utils.pitch import (LANDMARKS, PITCH_LENGTH, PITCH_WIDTH, draw_pitch, draw_grid,
                          compute_homography, project_point)

st.set_page_config(page_title="Vista Tattica dall'Alto", page_icon="⚽", layout="wide")

# st.fragment (stabile da Streamlit 1.37) fa sì che i clic nell'area di editing
# rifacciano girare solo quella porzione di pagina, non l'intero script — è questo
# il cambiamento che rende l'interazione reattiva. Fallback per versioni più vecchie.
FRAGMENT = getattr(st, "fragment", None) or st.experimental_fragment

DISPLAY_MAX_W = 980
TEAM_COLORS = {0: (224, 102, 43), 1: (36, 97, 201), -1: (138, 138, 128)}  # RGB
TEAM_COLORS_HEX = {0: "#e0662b", 1: "#2461c9", -1: "#8a8a80"}
TEAM_LABELS = {0: "Squadra A", 1: "Squadra B", -1: "Arbitro/altro"}
CALIB_COLOR = (255, 47, 208)
BALL_COLOR = (255, 255, 255)
PENDING_COLOR = (255, 200, 0)
DELETE_COLOR = (235, 60, 60)

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
        "frame": None,             # np.array BGR, risoluzione originale
        "base_img": None,          # PIL.Image già ridimensionata, senza annotazioni (cache manuale)
        "img_scale": 1.0,

        "players": [],              # lista definitiva: dict id/bbox/x/y/team
        "ball": None,                # dict definitivo {"x":.., "y":..} oppure None
        "calib_points": [],          # lista di dict img/world/label

        # --- bozza: modifiche fatte ma non ancora "Applicate" ---
        "player_edits": {},          # {player_id: nuova_squadra}
        "players_to_delete": set(),  # id di giocatori definitivi segnati per eliminazione
        "new_players_draft": [],     # nuovi giocatori in attesa: [{id, x, y, team}]
        "ball_draft": None,          # None = nessuna modifica; "REMOVE"; oppure {"x","y"}
        "next_temp_id": -1,

        "pending_click": None,       # ultimo clic in coordinate immagine originale (modo calib)
        "last_raw_click": None,      # per deduplicare i clic del componente
        "selected_player_id": None,
        "next_player_id": 0,
        "detected_preview": None,    # risultato rilevamento cache, in attesa di "Applica alla lista"

        "calib_dirty": True,         # True se la vista 2D va ricalcolata
        "view": None,                # ultima vista 2D applicata: {H, errors, rows, png, settings}

        # --- opzioni di visualizzazione della vista 2D ---
        "team_filter": "all",        # "all" | "A" | "B"
        "show_grid": False,
        "grid_h": [round(PITCH_WIDTH * i / 6, 1) for i in range(1, 6)],   # 5 linee orizzontali
        "grid_v": [round(PITCH_LENGTH * i / 5, 1) for i in range(1, 5)],  # 4 linee verticali
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def mark_dirty():
    st.session_state.calib_dirty = True


def set_frame(frame):
    """Imposta un nuovo frame sorgente e precalcola l'immagine di base ridimensionata
    (fatto una sola volta qui, non ad ogni clic, per restare reattivi)."""
    h, w = frame.shape[:2]
    scale = min(1.0, DISPLAY_MAX_W / w)
    disp_w, disp_h = int(w * scale), int(h * scale)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.frame = frame
    st.session_state.base_img = Image.fromarray(rgb).resize((disp_w, disp_h))
    st.session_state.img_scale = scale
    st.session_state.players = []
    st.session_state.ball = None
    st.session_state.calib_points = []
    st.session_state.detected_preview = None
    st.session_state.view = None
    st.session_state.calib_dirty = True
    discard_all_pending()


def effective_players():
    """Lista "vista corrente" dei giocatori: definitivi + bozze di modifica, per
    disegno e selezione. Non tocca lo stato definitivo."""
    items = []
    for p in st.session_state.players:
        items.append({
            "id": p["id"], "x": p["x"], "y": p["y"],
            "team": st.session_state.player_edits.get(p["id"], p["team"]),
            "pending_edit": p["id"] in st.session_state.player_edits,
            "pending_delete": p["id"] in st.session_state.players_to_delete,
            "is_draft": False,
        })
    for d in st.session_state.new_players_draft:
        items.append({
            "id": d["id"], "x": d["x"], "y": d["y"], "team": d["team"],
            "pending_edit": False, "pending_delete": False, "is_draft": True,
        })
    return items


def pending_count():
    n = len(st.session_state.player_edits) + len(st.session_state.players_to_delete) \
        + len(st.session_state.new_players_draft)
    if st.session_state.ball_draft is not None:
        n += 1
    return n


def apply_all_pending():
    new_list = []
    for p in st.session_state.players:
        if p["id"] in st.session_state.players_to_delete:
            continue
        if p["id"] in st.session_state.player_edits:
            p = dict(p)
            p["team"] = st.session_state.player_edits[p["id"]]
        new_list.append(p)
    for d in st.session_state.new_players_draft:
        new_list.append({
            "id": st.session_state.next_player_id, "bbox": None, "conf": None,
            "x": d["x"], "y": d["y"], "team": d["team"] if d["team"] is not None else -1,
        })
        st.session_state.next_player_id += 1
    st.session_state.players = new_list

    if st.session_state.ball_draft is not None:
        st.session_state.ball = None if st.session_state.ball_draft == "REMOVE" else st.session_state.ball_draft

    st.session_state.player_edits = {}
    st.session_state.players_to_delete = set()
    st.session_state.new_players_draft = []
    st.session_state.ball_draft = None
    st.session_state.selected_player_id = None
    mark_dirty()


def discard_all_pending():
    st.session_state.player_edits = {}
    st.session_state.players_to_delete = set()
    st.session_state.new_players_draft = []
    st.session_state.ball_draft = None
    st.session_state.selected_player_id = None


# ----------------------------------------------------------------------
# Utility di disegno
# ----------------------------------------------------------------------
def render_display_image(mode, base_img, scale, players_eff, ball, ball_draft,
                          calib_points, pending_click, selected_id):
    """Copia l'immagine di base (già ridimensionata, in cache) e disegna sopra solo
    le annotazioni leggere: niente resize/conversione colore ad ogni clic."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    for it in players_eff:
        x, y = it["x"] * scale, it["y"] * scale
        r = 7 if mode == MODE_PLAYERS else 5
        if it["pending_delete"]:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=DELETE_COLOR, width=3)
            draw.line([x - r, y - r, x + r, y + r], fill=DELETE_COLOR, width=2)
            draw.line([x - r, y + r, x + r, y - r], fill=DELETE_COLOR, width=2)
        else:
            color = TEAM_COLORS.get(it["team"], TEAM_COLORS[-1]) if it["team"] is not None else (170, 170, 170)
            outline = PENDING_COLOR if (it["pending_edit"] or it["is_draft"]) else (0, 0, 0)
            width = 3 if outline == PENDING_COLOR else 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=outline, width=width)
        if mode == MODE_PLAYERS and it["id"] == selected_id:
            draw.ellipse([x - r - 5, y - r - 5, x + r + 5, y + r + 5], outline=(255, 255, 0), width=3)

    if ball is not None and ball_draft != "REMOVE":
        x, y = ball["x"] * scale, ball["y"] * scale
        r = 8 if mode == MODE_BALL else 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill=BALL_COLOR, outline=(0, 0, 0), width=2)
    elif ball is not None and ball_draft == "REMOVE":
        x, y = ball["x"] * scale, ball["y"] * scale
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], outline=DELETE_COLOR, width=3)
    if isinstance(ball_draft, dict):
        x, y = ball_draft["x"] * scale, ball_draft["y"] * scale
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=PENDING_COLOR, width=3)
        if mode == MODE_BALL:
            draw.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 255, 0), width=3)

    if mode == MODE_CALIB:
        for i, c in enumerate(calib_points):
            x, y = c["img"][0] * scale, c["img"][1] * scale
            draw.line([x - 8, y, x + 8, y], fill=CALIB_COLOR, width=3)
            draw.line([x, y - 8, x, y + 8], fill=CALIB_COLOR, width=3)
            draw.text((x + 10, y - 10), str(i + 1), fill=CALIB_COLOR)
        if pending_click is not None:
            x, y = pending_click[0] * scale, pending_click[1] * scale
            draw.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 255, 255), width=3)

    return img


def nearest_item(items, x, y, max_dist=28):
    best, best_d = None, max_dist
    for it in items:
        d = ((it["x"] - x) ** 2 + (it["y"] - y) ** 2) ** 0.5
        if d < best_d:
            best, best_d = it, d
    return best


def current_view_settings():
    return {
        "team_filter": st.session_state.team_filter,
        "show_grid": st.session_state.show_grid,
        "grid_h": tuple(st.session_state.grid_h),
        "grid_v": tuple(st.session_state.grid_v),
    }


def build_view(calib_points, players, ball, settings):
    """Calcola omografia + disegna la vista 2D. Ritorna un dict con H, errors, rows, png (bytes)
    oppure None se la calibrazione non è ancora valida."""
    H, errors = compute_homography(calib_points)
    if H is None:
        return None

    team_filter = settings["team_filter"]
    if team_filter == "A":
        players = [p for p in players if p["team"] == 0]
    elif team_filter == "B":
        players = [p for p in players if p["team"] == 1]

    fig, ax = plt.subplots(figsize=(11, 7.2))
    draw_pitch(ax)
    if settings["show_grid"]:
        draw_grid(ax, settings["grid_h"], settings["grid_v"])

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

    return {"H": H, "errors": errors, "rows": rows, "png": png_buf.getvalue(), "settings": settings}


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
                    set_frame(first_frame)
                    st.success(f"Primo frame estratto dal video ({first_frame.shape[1]}x{first_frame.shape[0]}).")
            elif suffix in IMAGE_EXT:
                frame = cv2.imread(tmp_path)
                if frame is None:
                    st.error("Non riesco a leggere questa immagine.")
                else:
                    set_frame(frame)
                    st.success(f"Immagine caricata ({frame.shape[1]}x{frame.shape[0]}).")

    st.divider()
    conf = st.slider("Soglia di confidenza rilevamento", 0.10, 0.80, 0.15, 0.05)

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
            st.session_state.detected_preview = None
            discard_all_pending()
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

@FRAGMENT
def editor_fragment():
    """Tutta l'interazione a clic (giocatori, palla, calibrazione) vive qui dentro:
    un clic rifà girare solo questa funzione, non l'intera pagina (sidebar inclusa)."""
    mode_label = st.radio("Modalità", ["🧍 Giocatori", "⚽ Palla", "📐 Calibrazione campo"], horizontal=True)
    mode = {"🧍 Giocatori": MODE_PLAYERS, "⚽ Palla": MODE_BALL, "📐 Calibrazione campo": MODE_CALIB}[mode_label]

    players_eff = effective_players()

    col_img, col_panel = st.columns([2.2, 1])

    with col_img:
        img_disp = render_display_image(
            mode, st.session_state.base_img, st.session_state.img_scale, players_eff,
            st.session_state.ball, st.session_state.ball_draft, st.session_state.calib_points,
            st.session_state.pending_click, st.session_state.selected_player_id,
        )
        click = streamlit_image_coordinates(img_disp, key=f"img_{mode}")

        if mode == MODE_PLAYERS:
            st.caption("Clicca su un giocatore per cambiarne la squadra o eliminarlo. Clicca su un punto vuoto "
                       "per aggiungerne uno mancante. Le modifiche restano in bozza (bordo giallo) finché non "
                       "premi **Applica tutte le modifiche** qui sotto.")
        elif mode == MODE_BALL:
            st.caption("Clicca sulla posizione della palla per posizionarla o spostarla (resta in bozza finché "
                       "non applichi).")
        else:
            st.caption("Clicca su un punto riconoscibile del campo, poi scegli a cosa corrisponde nel pannello a destra.")

    # gestisci il clic (deduplicato rispetto all'ultimo elaborato) — operazioni leggere, solo bozza;
    # rimanendo dentro il fragment, rifanno girare solo questa funzione, non l'intera pagina
    if click is not None and click != st.session_state.last_raw_click:
        st.session_state.last_raw_click = click
        ox, oy = click["x"] / st.session_state.img_scale, click["y"] / st.session_state.img_scale

        if mode == MODE_PLAYERS:
            hit = nearest_item(players_eff, ox, oy)
            if hit is not None:
                st.session_state.selected_player_id = hit["id"]
            else:
                new_id = st.session_state.next_temp_id
                st.session_state.next_temp_id -= 1
                st.session_state.new_players_draft.append({"id": new_id, "x": ox, "y": oy, "team": None})
                st.session_state.selected_player_id = new_id
        elif mode == MODE_BALL:
            st.session_state.ball_draft = {"x": ox, "y": oy}
        else:
            st.session_state.pending_click = (ox, oy)
        st.rerun()

    # ------------------------------------------------------------------
    # Pannello laterale contestuale
    # ------------------------------------------------------------------
    with col_panel:
        if mode == MODE_PLAYERS:
            sel = st.session_state.selected_player_id
            eff = next((it for it in players_eff if it["id"] == sel), None) if sel is not None else None
            if eff is None:
                st.session_state.selected_player_id = None
                st.info("Nessun giocatore selezionato.")
            elif eff["is_draft"]:
                st.warning("🆕 Nuovo giocatore in attesa di conferma.")
                team_label = st.selectbox("Squadra", list(TEAM_LABELS.values()), key=f"team_new_{sel}")
                draft = next(d for d in st.session_state.new_players_draft if d["id"] == sel)
                draft["team"] = {v: k for k, v in TEAM_LABELS.items()}[team_label]
                if st.button("🗑️ Rimuovi questo nuovo giocatore", use_container_width=True):
                    st.session_state.new_players_draft = [
                        d for d in st.session_state.new_players_draft if d["id"] != sel]
                    st.session_state.selected_player_id = None
                    st.rerun()
            else:
                status_bits = []
                if eff["pending_edit"]:
                    status_bits.append("modifica squadra in attesa")
                if eff["pending_delete"]:
                    status_bits.append("eliminazione in attesa")
                status = f"  ⏳ _{', '.join(status_bits)}_" if status_bits else ""
                st.markdown(f"**Giocatore #{eff['id']}** — squadra: {TEAM_LABELS[eff['team']]}{status}")
                c1, c2 = st.columns(2)
                if c1.button("🟠 Squadra A", use_container_width=True):
                    st.session_state.player_edits[eff["id"]] = 0; st.rerun()
                if c2.button("🔵 Squadra B", use_container_width=True):
                    st.session_state.player_edits[eff["id"]] = 1; st.rerun()
                c3, c4 = st.columns(2)
                if c3.button("⚪ Arbitro/altro", use_container_width=True):
                    st.session_state.player_edits[eff["id"]] = -1; st.rerun()
                del_label = "↩️ Annulla eliminazione" if eff["pending_delete"] else "🗑️ Elimina"
                if c4.button(del_label, use_container_width=True):
                    if eff["pending_delete"]:
                        st.session_state.players_to_delete.discard(eff["id"])
                    else:
                        st.session_state.players_to_delete.add(eff["id"])
                    st.rerun()

        elif mode == MODE_BALL:
            draft = st.session_state.ball_draft
            committed = st.session_state.ball
            if draft == "REMOVE":
                st.warning("🗑️ Eliminazione della palla in attesa di conferma.")
                if st.button("↩️ Annulla eliminazione", use_container_width=True):
                    st.session_state.ball_draft = None
                    st.rerun()
            elif isinstance(draft, dict):
                st.warning(f"🆕 Nuova posizione in attesa: ({draft['x']:.0f}, {draft['y']:.0f})")
                if st.button("✕ Annulla questa modifica", use_container_width=True):
                    st.session_state.ball_draft = None
                    st.rerun()
            elif committed is not None:
                st.markdown(f"**Palla** — posizione attuale: ({committed['x']:.0f}, {committed['y']:.0f})")
                if st.button("🗑️ Rimuovi palla", use_container_width=True):
                    st.session_state.ball_draft = "REMOVE"
                    st.rerun()
            else:
                st.info("Palla non ancora posizionata. Clicca sull'immagine a sinistra.")

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

    # ------------------------------------------------------------------
    # Barra "modifiche in sospeso" — dentro il fragment così resta rapida,
    # tranne "Applica" che forza un rerun di tutta la pagina: deve aggiornare
    # anche il contatore in sidebar e la sezione Vista 2D qui sotto.
    # ------------------------------------------------------------------
    n_pending = pending_count()
    if n_pending > 0:
        st.divider()
        bar_l, bar_r = st.columns([3, 2])
        bar_l.warning(f"✏️ **{n_pending}** modifica/che in attesa — non ancora definitive.")
        with bar_r:
            ca, cb = st.columns(2)
            if ca.button("✅ Applica tutte le modifiche", type="primary", use_container_width=True):
                apply_all_pending()
                st.rerun(scope="app")
            if cb.button("✕ Scarta modifiche", use_container_width=True):
                discard_all_pending()
                st.rerun()


editor_fragment()


# ----------------------------------------------------------------------
# Vista 2D + esportazione
# ----------------------------------------------------------------------
st.divider()
st.header("Vista Tattica dall'Alto — proiezione 2D")

n_calib = len(st.session_state.calib_points)
if n_calib < 4:
    st.warning(f"Servono almeno 4 punti di calibrazione non allineati (attuali: {n_calib}).")
else:
    with st.expander("⚙️ Opzioni vista 2D (squadra da mostrare, griglia)"):
        filter_label = st.radio(
            "Giocatori da mostrare",
            ["Tutti", "Solo Squadra A", "Solo Squadra B"],
            index={"all": 0, "A": 1, "B": 2}[st.session_state.team_filter],
            horizontal=True,
        )
        st.session_state.team_filter = {"Tutti": "all", "Solo Squadra A": "A", "Solo Squadra B": "B"}[filter_label]

        st.session_state.show_grid = st.checkbox("Mostra griglia (scacchiera) sul campo",
                                                  value=st.session_state.show_grid)
        if st.session_state.show_grid:
            st.caption("5 linee orizzontali e 4 linee verticali, regolabili singolarmente (non devono "
                       "essere equidistanti).")
            gh_cols = st.columns(5)
            for i, col in enumerate(gh_cols):
                st.session_state.grid_h[i] = col.slider(
                    f"Orizz. {i+1}", 0.0, PITCH_WIDTH, st.session_state.grid_h[i], 0.5, key=f"grid_h_{i}")
            gv_cols = st.columns(4)
            for i, col in enumerate(gv_cols):
                st.session_state.grid_v[i] = col.slider(
                    f"Vert. {i+1}", 0.0, PITCH_LENGTH, st.session_state.grid_v[i], 0.5, key=f"grid_v_{i}")

    settings = current_view_settings()
    settings_changed = st.session_state.view is None or st.session_state.view.get("settings") != settings
    if st.session_state.calib_dirty or settings_changed:
        st.warning("⚠️ Punti, giocatori, palla o opzioni di vista modificati dall'ultimo aggiornamento.")
    apply_label = "🔄 Applica calibrazione / aggiorna vista 2D" if st.session_state.view is not None \
        else "📐 Genera vista 2D"
    if st.button(apply_label, type="primary"):
        result = build_view(st.session_state.calib_points, st.session_state.players, st.session_state.ball, settings)
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
