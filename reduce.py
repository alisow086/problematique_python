import numpy as np


def reduce(I_source, LIGNES, COLONNES):
    """Resize to LIGNES x COLONNES by bilinear interpolation, up or down.

    Corner-aligned convention: the outermost output pixels land exactly on the
    outermost input pixels.
    """
    I_source = np.asarray(I_source, dtype=float)
    hauteur, largeur = I_source.shape

    # Guard against division by zero on a single-pixel target
    echelle_l = (hauteur - 1) / (LIGNES - 1) if LIGNES > 1 else 0.0
    echelle_c = (largeur - 1) / (COLONNES - 1) if COLONNES > 1 else 0.0

    I_reduced = np.zeros((LIGNES, COLONNES), dtype=float)

    for l_out in range(LIGNES):
        for c_out in range(COLONNES):
            # Fractional coordinates in the source image
            l_s = l_out * echelle_l
            c_s = c_out * echelle_c

            l0 = int(np.floor(l_s))
            c0 = int(np.floor(c_s))
            l1 = min(l0 + 1, hauteur - 1)
            c1 = min(c0 + 1, largeur - 1)

            dl = l_s - l0
            dc = c_s - c0

            # Weighted average of the four surrounding neighbours
            I_reduced[l_out, c_out] = (
                (1 - dl) * (1 - dc) * I_source[l0, c0]
                + (1 - dl) * dc * I_source[l0, c1]
                + dl * (1 - dc) * I_source[l1, c0]
                + dl * dc * I_source[l1, c1]
            )

    # 8-bit output. floor(x + 0.5), not np.round, which rounds half to even.
    return np.floor(I_reduced + 0.5)
