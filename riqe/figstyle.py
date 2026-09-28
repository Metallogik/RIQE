"""Stile delle figure, con le regole seguite dichiarate.

La tavolozza e' l'istanza di riferimento validata: gli slot sono assegnati
**in ordine fisso, mai ciclati**.  Le figure dell'articolo sono per la stampa,
quindi solo modalita' chiara.

Regole rispettate, non negoziabili:

  * mai due assi y nella stessa figura.  Quando servono due grandezze di scala
    diversa sullo stesso asse x -- il caso della figura principale, punteggio
    e ritenzione del segnale -- si usano **due pannelli impilati che
    condividono la x**, non un secondo asse.
  * l'identita' di una serie non e' mai affidata al solo colore: ogni serie
    porta anche tratteggio e marcatore propri, piu' etichetta diretta quando
    le serie sono poche.  Serve ai lettori con deficit di visione dei colori e
    alla stampa in bianco e nero.
  * griglia e assi recessivi, marcatori sottili, nessun numero su ogni punto.
  * sequenziale = una sola tinta chiaro->scuro; divergente = due tinte con
    grigio neutro al centro.  Mai arcobaleno.

La tavolozza categorica supera le soglie CVD sulla lista di coppie
**adiacenti** fino a cinque slot; oltre i tre slot non e' valida per forme a
coppie-tutte (dispersione, bolle), dove si accorpa in "altro" o si sfaccetta.
"""

from __future__ import annotations

#: tavolozza categorica, ordine fisso
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948")
#: codifica secondaria, appaiata agli slot
DASHES = ((), (5, 2), (1, 1.6), (7, 2, 1.5, 2), (3, 1.5, 1, 1.5),
          (6, 3), (2, 2, 6, 2), (9, 2))
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

#: rampa sequenziale a una tinta (blu), chiaro -> scuro
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
#: divergente, poli caldo/freddo con grigio neutro al centro
DIVERGING = ("#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f19b9a", "#e34948", "#8f2322")

SURFACE = "#ffffff"  # bianco: le figure sono per la stampa
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2de"


def apply(mpl) -> None:
    """Applica lo stile a matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def style_for(i: int) -> dict:
    """Colore, tratteggio e marcatore dello slot i, in ordine fisso."""
    j = i % len(SERIES)
    d = DASHES[j]
    out = {"color": SERIES[j], "marker": MARKERS[j]}
    if d:
        out["dashes"] = d
    return out


def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
