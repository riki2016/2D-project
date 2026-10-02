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
LANDMARKS = {
    # --- centro campo ---
    "Centro del campo (centro del cerchio di centrocampo)": (52.5, 34),
    "Dischetto di rigore — lato SINISTRO dello schermo": (11, 34),
    "Dischetto di rigore — lato DESTRO dello schermo": (94, 34),

    # --- perimetro del campo: i 4 angoli, nominati come li vedi sullo schermo ---
    "Angolo campo — ALTO SINISTRA": (0, 0),
    "Angolo campo — ALTO DESTRA": (105, 0),
    "Angolo campo — BASSO SINISTRA": (0, 68),
    "Angolo campo — BASSO DESTRA": (105, 68),

    # --- area di rigore sinistra: solo angoli ---
    "Area rigore SINISTRA — angolo ALTO, sulla linea di fondo": (0, 13.84),
    "Area rigore SINISTRA — angolo ALTO, bordo esterno area": (16.5, 13.84),
    "Area rigore SINISTRA — angolo BASSO, bordo esterno area": (16.5, 54.16),
    "Area rigore SINISTRA — angolo BASSO, sulla linea di fondo": (0, 54.16),

    # --- area di rigore destra: solo angoli ---
    "Area rigore DESTRA — angolo ALTO, sulla linea di fondo": (105, 13.84),
    "Area rigore DESTRA — angolo ALTO, bordo esterno area": (88.5, 13.84),
    "Area rigore DESTRA — angolo BASSO, bordo esterno area": (88.5, 54.16),
    "Area rigore DESTRA — angolo BASSO, sulla linea di fondo": (105, 54.16),

    # --- in alternativa ai 4 angoli: dove il bordo esterno dell'area incrocerebbe
    # la linea laterale se prolungato — utile se vedi la laterale e il bordo area
    # ma non l'angolo vero e proprio (es. inquadratura stretta) ---
    "Area rigore SINISTRA — bordo area ∩ linea laterale ALTA": (16.5, 0),
    "Area rigore SINISTRA — bordo area ∩ linea laterale BASSA": (16.5, 68),
    "Area rigore DESTRA — bordo area ∩ linea laterale ALTA": (88.5, 0),
    "Area rigore DESTRA — bordo area ∩ linea laterale BASSA": (88.5, 68),

    "Punto personalizzato (coordinate X,Y mondo a mano)": None,
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
