"""Rilevamento giocatori (YOLOv8) e stima della squadra per colore maglia (K-Means)."""
import cv2
import numpy as np
from sklearn.cluster import KMeans


def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def detect_players(model, frame, conf=0.30, imgsz=1280):
    """Ritorna una lista di dict: id, bbox [x1,y1,x2,y2], conf, x, y (punto a terra), team (None)."""
    results = model.predict(frame, conf=conf, imgsz=imgsz, classes=[0], verbose=False)
    r = results[0]
    players = []
    if r.boxes is None or len(r.boxes) == 0:
        return players
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    for i, (box, c) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = box.tolist()
        players.append({
            "id": i,
            "bbox": [x1, y1, x2, y2],
            "conf": float(c),
            "x": (x1 + x2) / 2,
            "y": y2,
            "team": None,
        })
    return players


def _dominant_jersey_color(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    torso = frame[y1 + int(h * 0.15): y1 + int(h * 0.55), x1:x2]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    px = hsv.reshape(-1, 3)
    mask = ~((px[:, 0] > 35) & (px[:, 0] < 85) & (px[:, 1] > 40))
    mask &= (px[:, 2] > 30) & (px[:, 2] < 240)
    f = px[mask]
    if len(f) < 10:
        f = px
    return np.median(f, axis=0)


def assign_teams(frame, players, n_teams=2):
    """Aggiorna players[i]['team'] in place (0, 1, o -1 se non classificabile) via K-Means
    sul colore mediano del torso di ciascun giocatore."""
    colors, idxs = [], []
    for i, p in enumerate(players):
        c = _dominant_jersey_color(frame, p["bbox"])
        if c is not None:
            colors.append(c)
            idxs.append(i)

    if len(colors) < n_teams:
        for p in players:
            p["team"] = -1
        return players

    km = KMeans(n_clusters=n_teams, n_init=10, random_state=42).fit(np.array(colors))
    for idx, lab in zip(idxs, km.labels_):
        players[idx]["team"] = int(lab)
    for p in players:
        if p["team"] is None:
            p["team"] = -1
    return players
