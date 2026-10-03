# Vista Tattica dall'Alto ⚽

App Streamlit per trasformare lo screenshot (o il primo frame di un video) di una partita
di calcio in una vista tattica 2D in scala reale: rilevamento giocatori, assegnazione
squadra per colore maglia, validazione manuale tramite clic diretto, calibrazione del
campo tramite clic diretto, e proiezione omografica sul campo regolamentare.

## Funzionalità

- **Caricamento** di un'immagine o di un video (in questo caso viene usato automaticamente
  il primo frame).
- **Rilevamento giocatori** con YOLOv8 (`ultralytics`) e **stima automatica della squadra**
  per colore maglia (K-Means sul colore mediano del torso, in spazio HSV). Il modello e il
  risultato del rilevamento sono tenuti in cache (`st.cache_resource`/`st.cache_data`): premendo
  di nuovo "Rileva giocatori" con la stessa immagine e soglia la risposta è immediata. Il
  risultato va poi confermato con **"✅ Applica alla lista"**, così non sovrascrive per sbaglio
  correzioni manuali già fatte.
- **Validazione manuale per clic diretto sull'immagine**: clicca su un giocatore per
  cambiargli squadra o eliminarlo; clicca su un punto vuoto per aggiungerne uno mancante.
- **Posizionamento della palla** (modalità "⚽ Palla"): clicca per posizionarla, un nuovo
  clic la sposta; viene proiettata anch'essa sulla vista 2D e inclusa nell'esportazione.
- **Calibrazione del campo per clic diretto**: clicca un punto riconoscibile dell'immagine e
  scegli dal menu a cosa corrisponde. L'elenco è volutamente essenziale: dischetto di rigore
  (sx/dx schermo), centrocampo, linea di centrocampo (incrocio alto/basso), i 4 corner e i
  4 angoli dell'area di rigore grande — tutti i nomi sono relativi a ciò che vedi sullo
  schermo, non alla porta di una squadra specifica. Funziona anche se il campo non è
  interamente visibile nel frame, bastano 4 punti non allineati.
- **Proiezione 2D in scala reale** tramite omografia (`cv2.findHomography`), con diagnostica
  dell'errore di riproiezione per punto. Anche qui il calcolo e il disegno della vista non
  avvengono ad ogni clic: si aggiornano solo premendo **"🔄 Applica calibrazione / aggiorna
  vista 2D"**, per restare reattivi durante la validazione.
- **Esportazione** dei risultati (giocatori + palla) in CSV, JSON e immagine PNG della vista
  tattica.

## Avvio in locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

Il primo rilevamento scarica automaticamente i pesi `yolov8n.pt` (modello leggero di
`ultralytics`); per maggiore accuratezza si può sostituire con `yolov8s.pt` o `yolov8m.pt`
in `utils/detection.py` (funzione `load_model`).

**Eseguirla in locale (sul tuo PC) invece che su un servizio cloud elimina la latenza di
rete ad ogni clic** e sfrutta la tua CPU/GPU per YOLO — è il modo più semplice per renderla
più reattiva senza toccare il codice.

### Perché l'interazione a clic è reattiva

Streamlit normalmente rifà girare l'intero script ad ogni interazione. Quest'app isola
l'area di clic (giocatori, palla, calibrazione) in un **fragment** (`@st.fragment`,
richiede `streamlit>=1.37`): un clic lì dentro rifà girare solo quella porzione di pagina,
non l'intera app (sidebar e upload inclusi). Il contatore in sidebar e la sezione "Vista 2D"
si aggiornano quando premi un pulsante "Applica" (che forza un aggiornamento completo) o
quando interagisci con qualcosa fuori dal fragment.

## Deploy su Streamlit Community Cloud

1. Pusha questo repository su GitHub.
2. Su [share.streamlit.io](https://share.streamlit.io) collega il repository e indica
   `app.py` come file principale.
3. Il file `requirements.txt` viene letto automaticamente per installare le dipendenze.

## Struttura del progetto

```
app.py                  interfaccia Streamlit (upload, clic, pannelli, export)
utils/
  detection.py           rilevamento YOLO + assegnazione squadra (K-Means)
  pitch.py                geometria del campo, disegno 2D, omografia
sample_data/             (opzionale) immagini/video di prova, esclusi da git
.streamlit/config.toml   tema dell'app
requirements.txt
```

## Note

- La qualità della calibrazione dipende dai punti scelti: preferisci punti ben distanti
  tra loro e non allineati sulla stessa retta, l'errore di riproiezione mostrato nel
  pannello "Qualità della calibrazione" aiuta a individuare i punti più imprecisi.
- L'assegnazione automatica della squadra può sbagliare in condizioni di luce o colori
  maglia simili: per questo è sempre possibile correggerla manualmente cliccando sul
  giocatore nella modalità "🧍 Giocatori".
