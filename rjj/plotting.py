import matplotlib as mpl

def anchored_text(ax, text, loc, fancybox=None, edgecolor=None, facecolor=None,
                  framealpha=None, frameon=None, pad=0, prop=None):
    if fancybox is None:
        fancybox = mpl.rcParams["legend.fancybox"] # default: True

    if facecolor is None:
        facecolor = mpl.rcParams["legend.facecolor"] # default: "inherit"
    if facecolor == "inherit":
        facecolor = mpl.rcParams["axes.facecolor"] # default: "white"

    if edgecolor is None:
        edgecolor = mpl.rcParams["legend.edgecolor"] # default: "0.8"
    if edgecolor == "inherit":
        edgecolor = mpl.rcParams["axes.edgecolor"]

    if framealpha is None:
        framealpha = mpl.rcParams["legend.framealpha"] # default: 0.8

    if frameon is None:
        frameon = mpl.rcParams["legend.frameon"] # default: True

    textbox = mpl.offsetbox.AnchoredText(text, loc, prop=prop)

    if fancybox:
        textbox.patch.set_boxstyle("round", pad=pad, rounding_size=0.2)
    else:
        textbox.patch.set_boxstyle("square", pad=pad)
    textbox.patch.set_facecolor(facecolor)
    textbox.patch.set_edgecolor(edgecolor)
    textbox.patch.set_alpha(framealpha)
    textbox.patch.set_visible(frameon)

    ax.add_artist(textbox)
    return textbox
