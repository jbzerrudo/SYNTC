"""
Shared figure furniture.

Every plotting script in this release takes `--titles`. It is OFF by default,
which is the setting the manuscript uses: a journal figure carries its title and
its explanation in the LaTeX \\caption, and burning the same words into the image
duplicates them, sets them in a font the journal did not choose, and makes them
uneditable at proof stage.

Turn it on when browsing a run folder, where a bare figure is hard to identify.
The plot is otherwise byte-identical, so the same command produces the figure in
the paper whether or not you looked at a titled version first.

Panel labels, (a) and (b), are NOT affected. Captions refer to them, so they
have to live in the image.
"""

TITLES = False


def title(fig, main, sub=None, x=0.012, y=1.03, ink="#0b0b0b",
          muted="#52514e"):
    """Figure title and one-line explanation, drawn only if --titles is set."""
    if not TITLES:
        return
    fig.suptitle(main, fontsize=13, color=ink, x=x, ha="left", y=y)
    if sub:
        fig.text(x, y - 0.05, sub, fontsize=8.5, color=muted, ha="left",
                 va="top")


def rect(default_top=0.94):
    """tight_layout rect: reserve space for the title only when it is drawn."""
    return [0, 0, 1, default_top] if TITLES else None


def tc_marker(turns=1.15, growth=0.62, start=0.20):
    """A tropical cyclone symbol as a matplotlib marker path.

    Two logarithmic spiral arms, drawn as strokes and used unfilled so the
    probability field underneath stays readable. That is the whole reason the
    genesis point was a plain dot before: a filled glyph large enough to be
    recognisable covers about two grid cells, hiding the highest-probability
    cells in the figure. An open symbol keeps the shape and lets the colour
    through.

    Arms turn counterclockwise, the northern-hemisphere convention. Pair it
    with a small centre dot for the eye, which is what marks the coordinate.
    """
    import numpy as np
    from matplotlib.path import Path
    v, c = [], []
    for sign in (1, -1):
        th = np.linspace(0.0, np.pi * turns, 60)
        r = start * np.exp(growth * th)
        x = sign * r * np.cos(th + np.pi / 2)
        y = sign * r * np.sin(th + np.pi / 2)
        v.extend(zip(x, y))
        c.extend([Path.MOVETO] + [Path.LINETO] * (len(th) - 1))
    v = np.asarray(v)
    v /= np.abs(v).max()          # unit box, so markersize means points
    return Path(v, c)
