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
