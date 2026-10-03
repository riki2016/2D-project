"""Geometria del campo regolamentare, disegno 2D e calcolo dell'omografia.

Tutte le misure sono in metri, convenzione FIFA per un campo 105 x 68 m.
L'origine (0,0) e l'orientamento (sinistra/destra, alto/basso) sono arbitrari:
l'importante è che i punti di calibrazione scelti dall'utente siano coerenti
con queste stesse coordinate (vedi LANDMARKS).
"""
import cv2
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Arc

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Tutti i nomi sono relativi a come appare il campo SUL TUO SCHERMO in questo frame
# (sinistra/destra, alto/basso dell'immagine) — non alla porta di una squadra o
# dell'altra. Basta guardare dove sei inquadrato e scegliere il punto corrispondente.
# Elenco volutamente ridotto ai soli punti richiesti.
LANDMARKS = {
    "Dischetto di rigore — lato SINISTRO dello schermo": (11, 34),
    "Dischetto di rigore — lato DESTRO dello schermo": (94, 34),
    "Centrocampo (centro del cerchio di centrocampo)": (52.5, 34),
    "Linea di centrocampo — incrocio ALTO (lato alto dello schermo)": (52.5, 0),
    "Linea di centrocampo — incrocio BASSO (lato basso dello schermo)": (52.5, 68),
    "Corner — ALTO DESTRA": (105, 0),
    "Corner — BASSO DESTRA": (105, 68),
    "Corner — ALTO SINISTRA": (0, 0),
    "Corner — BASSO SINISTRA": (0, 68),
    "Area di rigore grande — angolo ALTO DESTRA": (88.5, 13.84),
    "Area di rigore grande — angolo BASSO DESTRA": (88.5, 54.16),
    "Area di rigore grande — angolo ALTO SINISTRA": (16.5, 13.84),
    "Area di rigore grande — angolo BASSO SINISTRA": (16.5, 54.16),
}


def draw_pitch(ax, bg_color="#0e2e18", line_color="#eafaf0"):
    """Disegna il campo regolamentare (linee, aree, cerchio, dischetti) su un asse matplotlib."""
    ax.set_facecolor(bg_color)
    ax.add_patch(patches.Rectangle((0, 0), PITCH_LENGTH, PITCH_WIDTH,
                                    fill=False, edgecolor=line_color, linewidth=2))
    ax.plot([52.5, 52.5], [0, PITCH_WIDTH], color=line_color, linewidth=2)
    ax.add_patch(patches.Circle((52.5, 34), 9.15, fill=False, edgecolor=line_color, linewidth=2))
    ax.add_patch(patches.Circle((52.5, 34), 0.3, color=line_color))

    def box(x0, y0, w, h):
        ax.add_patch(patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=line_color, linewidth=2))

    box(0, 13.84, 16.5, 40.32)
    box(PITCH_LENGTH - 16.5, 13.84, 16.5, 40.32)
    box(0, 24.84, 5.5, 18.32)
    box(PITCH_LENGTH - 5.5, 24.84, 5.5, 18.32)

    ax.add_patch(patches.Circle((11, 34), 0.3, color=line_color))
    ax.add_patch(patches.Circle((94, 34), 0.3, color=line_color))

    ax.add_patch(Arc((11, 34), 2 * 9.15, 2 * 9.15, angle=0, theta1=-52.6, theta2=52.6,
                      color=line_color, linewidth=2))
    ax.add_patch(Arc((94, 34), 2 * 9.15, 2 * 9.15, angle=0, theta1=180 - 52.6, theta2=180 + 52.6,
                      color=line_color, linewidth=2))

    for cx, cy, a1, a2 in [(0, 0, 0, 90), (105, 0, 90, 180), (0, 68, 270, 360), (105, 68, 180, 270)]:
        ax.add_patch(Arc((cx, cy), 2, 2, angle=0, theta1=a1, theta2=a2, color=line_color, linewidth=1.5))

    ax.plot([0, 0], [30.34, 30.34 + 7.32], color=line_color, linewidth=4)
    ax.plot([105, 105], [30.34, 30.34 + 7.32], color=line_color, linewidth=4)

    ax.set_xlim(-4, PITCH_LENGTH + 4)
    ax.set_ylim(-4, PITCH_WIDTH + 4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")


def draw_grid(ax, h_lines, v_lines, color="#ffd23f", alpha=0.55):
    """Disegna una griglia regolabile ("scacchiera") sul campo: h_lines sono posizioni Y
    (linee orizzontali, 0-68 m), v_lines sono posizioni X (linee verticali, 0-105 m).
    Ogni linea è indipendente dalle altre, non è richiesta una spaziatura regolare."""
    for y in h_lines:
        ax.plot([0, PITCH_LENGTH], [y, y], color=color, linewidth=1.3, linestyle="--", alpha=alpha, zorder=4)
    for x in v_lines:
        ax.plot([x, x], [0, PITCH_WIDTH], color=color, linewidth=1.3, linestyle="--", alpha=alpha, zorder=4)


def compute_homography(calib_points):
    """calib_points: lista di dict {"img": (x,y), "world": (X,Y)}.
    Ritorna (H, errori_di_riproiezione) oppure (None, None) se non calcolabile."""
    if len(calib_points) < 4:
        return None, None
    src = np.array([c["img"] for c in calib_points], dtype=np.float32)
    dst = np.array([c["world"] for c in calib_points], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        return None, None
    reproj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(reproj - dst, axis=1)
    return H, errors


def project_point(H, x, y):
    pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), H)
    return float(pt[0, 0, 0]), float(pt[0, 0, 1])
