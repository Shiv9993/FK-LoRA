"""One look for every figure of the paper.

Import this module in every script that draws a figure and use its names; do not
write colours or font sizes in the scripts themselves.

    import figstyle as fs
    fs.apply()                       # sets matplotlib rcParams
    ax.plot(x, y, color=fs.color("FinderLoRA"), marker=fs.marker("FinderLoRA"))
    fs.save(fig, "answer_grid")      # writes figures/answer_grid.pdf and .png

Rules behind the choices (checked with the palette validator of the data-viz
method: lightness band, chroma, colour-blind separation of neighbours; three of the
hues are light on white, so every series also gets its own marker and a direct
label or a legend entry):
  * a colour belongs to a method, in every figure, and never to its rank;
  * our methods are green, charcoal and gold, the published methods vermilion and
    crimson, simple baselines are grey; no blue and no purple;
  * magnitude (heat maps) is one hue, light to dark; never a rainbow;
  * text is always ink-coloured, never in the colour of a series.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))

# ink and structure
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a86"
AXIS, GRID, SURFACE, PANEL = "#52514e", "#dddcd7", "#ffffff", "#f6f5f1"

# method colours (fixed for the whole paper). No blue and no purple, at the author's wish:
# green, gold, vermilion and crimson (checked together with the palette validator for
# colour-blind separation of every pair), plus charcoal and greys.
GREEN, GOLD, VERMILION, CRIMSON = "#1f8f5f", "#d1a800", "#e0561f", "#9e1f3a"
CHARCOAL, GREY, LIGHTGREY, TAUPE = "#2f2f2c", "#8a8a86", "#c9c8c2", "#7a6a58"
DARKGREY = "#52514e"
LIGHTGREEN = "#9fcdb0"

_METHODS = {
    # name                colour     marker
    "FinderLoRA":        (GREEN,     "o"),
    "KeeperLoRA":        (CHARCOAL,  "D"),
    "ClaimLoRA":         (GOLD,      "s"),
    "InfLoRA":           (VERMILION, "^"),
    "GainLoRA":          (CRIMSON,   "v"),
    "Plain adapters":    (GREY,      "X"),
    "Exact key":         (LIGHTGREY, "P"),
    "Nearest zone":      (TAUPE,     "h"),
    "Nearest centre":    (TAUPE,     "*"),
    "Exact claims first": (CHARCOAL, "D"),
}
_ALIASES = {
    "ClaimLoRA (ours)": "ClaimLoRA", "FinderLoRA (ours)": "FinderLoRA", "KeeperLoRA (ours)": "KeeperLoRA",
    "Exact claims first (variant)": "Exact claims first",
}

# state colours for "kept" against "changed" pictures
KEPT, CHANGED = "#e3e6df", "#d9481c"      # pale grey-green: answer kept; vermilion: answer changed
EMPTY = "#f1f0ec"                          # a cell with no value

# one-hue ramp for magnitude (0 = near white, high = dark green)
SEQ = LinearSegmentedColormap.from_list(
    "paper_green", ["#f5f9f5", "#d5e8da", "#9fcdb0", "#55a67f", "#1f7a52", "#0d4a31"])
SEQ.set_bad(EMPTY)


def _key(name):
    return _ALIASES.get(name, name)


def color(name):
    return _METHODS[_key(name)][0]


def marker(name):
    return _METHODS[_key(name)][1]


def ink_on(rgb_or_value_fraction):
    """Ink colour for a number printed on a cell of the SEQ ramp (fraction 0..1)."""
    return "#ffffff" if rgb_or_value_fraction >= 0.55 else INK


def apply():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 5,
        "axes.labelsize": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 2.5, "ytick.major.size": 0, "xtick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 5, "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 0.7,
        "legend.fontsize": 8, "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def panel_title(ax, letter, text):
    """Panel heading: bold letter, plain text, at the left."""
    ax.set_title("")
    ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=INK)
    ax.text(0.0, 1.03, f"      {text}", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK)


def note(ax, text, xy, xytext, ha="left", va="center", color=INK2, arrow=True):
    """A short remark with a thin arrow to the place it is about (data coordinates)."""
    props = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=2) if arrow else None
    return ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=7.5, color=color, arrowprops=props)


def save(fig, name, dpi=220):
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
# additions (figure redesign): task colours, direct labels, legend above, dumbbell marks
# ---------------------------------------------------------------------------------------
# five tasks in the order they were learned: slate steps, light to dark, not a method colour
TASKS = ["#5b4636", "#7a8b2a", "#c9a227", "#c0623a", "#9a9a96"]   # brown, olive, gold, terracotta, grey
TASK_MARKS = ["o", "s", "^", "D", "v"]

# three kinds of edit (not methods): charcoal, terracotta, gold
EDITS = {"typos": "#2f2f2c", "dropped words": "#c0623a", "replaced words": "#c9a227"}


def spread(ys, gap, lo=None, hi=None):
    """Move label heights apart so that neighbours are at least `gap` apart (same order as the data)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = [float(y) for y in ys]
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    # centre the pushed groups on their original mean so labels stay close to their lines
    shift = (sum(ys) - sum(out)) / len(ys)
    out = [y + shift for y in out]
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    if hi is not None and max(out) > hi:
        d = max(out) - hi
        out = [y - d for y in out]
    if lo is not None and min(out) < lo:
        d = lo - min(out)
        out = [y + d for y in out]
    return out


def end_label(ax, x, y, text, dx=6, size=8, **kw):
    """Name at the end of a line, in ink; dx is the gap in points."""
    return ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points", ha="left", va="center",
                       fontsize=size, color=INK, annotation_clip=False, **kw)


def legend_above(fig, handles, labels, ncol=None, y=1.0, **kw):
    """One compact legend centred above the panels."""
    return fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(labels),
                      handlelength=1.4, columnspacing=1.3, handletextpad=0.5, borderaxespad=0.0, **kw)


def open_filled_handles(color=None):
    """Legend handles for the dumbbell charts: open circle, filled circle (grey ink unless a colour is given)."""
    from matplotlib.lines import Line2D
    c = color or INK2
    return [Line2D([], [], ls="none", marker="o", ms=5.5, mfc="white", mec=c, mew=1.2),
            Line2D([], [], ls="none", marker="o", ms=5.5, mfc=c, mec=c, mew=1.2)]


def dumbbell(ax, y, a, b, col, lw=2.4, ms=5.5, z=3):
    """Thick line in the method colour between an open circle (a) and a filled circle (b)."""
    ax.plot([a, b], [y, y], color=col, lw=lw, solid_capstyle="round", zorder=z)
    ax.plot(a, y, ls="none", marker="o", ms=ms, mfc="white", mec=col, mew=1.3, zorder=z + 1)
    ax.plot(b, y, ls="none", marker="o", ms=ms, mfc=col, mec=col, mew=1.3, zorder=z + 2)
