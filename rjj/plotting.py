import matplotlib as mpl

def anchored_text(ax, text, loc, style="square"):
    textbox = mpl.offsetbox.AnchoredText(text, loc)
    if style == "square":
        textbox.patch.set_boxstyle("square")
    elif style == "round":
        textbox.patch.set_boxstyle("round", pad=0, rounding_size=0.2)
    elif style == "legend":
        textbox.patch.set_boxstyle("round", pad=0, rounding_size=0.2)
        textbox.patch.set_facecolor((1.0, 1.0, 1.0, 0.8))
        textbox.patch.set_edgecolor((0.8, 0.8, 0.8, 0.8))
    ax.add_artist(textbox)
    return textbox
