import numpy as np


def reduce(I_source, LIGNES, COLONNES):
    # reimensionnement par interpolation lineaire
    I_source = np.asarray(I_source, dtype=float)
    hauteur, largeur = I_source.shape

    # Facteurs d'échelle, avec garde contre la division par zéro
    echelle_l = (hauteur - 1) / (LIGNES - 1) if LIGNES > 1 else 0.0
    echelle_c = (largeur - 1) / (COLONNES - 1) if COLONNES > 1 else 0.0

    I_reduced = np.zeros((LIGNES, COLONNES), dtype=float)

    for l_out in range(LIGNES):
        for c_out in range(COLONNES):
            # Coordonnées fractionnaires correspondantes dans l'image source
            l_s = l_out * echelle_l
            c_s = c_out * echelle_c

            # Les quatre voisins entiers qui encadrent ce point
            l0 = int(np.floor(l_s))
            c0 = int(np.floor(c_s))
            l1 = min(l0 + 1, hauteur - 1)
            c1 = min(c0 + 1, largeur - 1)

            # Distances fractionnaires au voisin supérieur gauche
            dl = l_s - l0
            dc = c_s - c0

            # Moyenne pondérée des quatre voisins
            I_reduced[l_out, c_out] = (
                (1 - dl) * (1 - dc) * I_source[l0, c0]
                + (1 - dl) * dc * I_source[l0, c1]
                + dl * (1 - dc) * I_source[l1, c0]
                + dl * dc * I_source[l1, c1]
            )

    # Profondeur 8 bits en sortie (spécification 1 du guide).
    # floor(x + 0.5) et non np.round, qui applique l'arrondi bancaire.
    return np.floor(I_reduced + 0.5)
