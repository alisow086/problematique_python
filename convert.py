import numpy as np

# Poids de luminance ITU-R BT.601 : l'oeil est beaucoup plus sensible
# au vert qu'au bleu, d'où des coefficients inégaux.
POIDS_LUMINANCE = np.array([0.299, 0.587, 0.114])


def convert(I_source):
    """Réduit la profondeur de couleur : RVB 24 bits -> niveaux de gris 8 bits.

    Accepte une image (H, W, 3) en RVB ou (H, W) déjà en niveaux de gris.
    Retourne toujours une matrice 2D.
    """
    I_source = np.asarray(I_source, dtype=float)

    # Image déjà en niveaux de gris 8 bits : rien à convertir
    if I_source.ndim == 2:
        return I_source

    # Image RVB 24 bits : combinaison linéaire pondérée des trois canaux
    # (le [:, :, :3] ignore un éventuel canal alpha)
    return I_source[:, :, :3] @ POIDS_LUMINANCE
