import numpy as np

# Transformée et format des métadonnées partagés avec le codeur : les deux
# doivent parcourir les coefficients dans exactement le même ordre.
from DCT_encode import (matrice_dct, recomposer_depuis_blocs,
                        FRACTION_PAS, OFFSET_MOYENNE)


def idct_2d_blocs(coefficients, T):
    """DCT 2D inverse de chaque bloc : bloc = T.T @ C @ T."""
    blocs = np.zeros(coefficients.shape)
    for b in range(len(coefficients)):
        blocs[b] = T.T @ coefficients[b] @ T
    return blocs


def DCT_decode(I_encoded, I_metadata, ArgumentY):
    """Décodeur par transformée en cosinus discrète.

    ArgumentY = (taille_bloc, lignes, colonnes).
    I_encoded est un dictionnaire {nombre de bits: indices quantifiés}.
    """
    N, lignes, colonnes = ArgumentY
    meta = np.asarray(I_metadata, dtype=int)

    # Les trois tables, dans l'ordre où le codeur les a concaténées
    B = meta[:N * N].reshape(N, N)
    pas_codes = meta[N * N:2 * N * N]
    m_codes = meta[2 * N * N:3 * N * N]

    nb_blocs = (lignes // N) * (colonnes // N)
    coefficients = np.zeros((nb_blocs, N, N))

    # Curseur de lecture dans chaque cellule : plusieurs coefficients peuvent
    # partager la même largeur de code et donc la même cellule.
    curseurs = {}
    for bits in I_encoded:
        curseurs[bits] = 0

    position = 0
    for i in range(N):
        for j in range(N):
            moyenne = m_codes[position] - OFFSET_MOYENNE
            bits = int(B[i, j])

            if bits == 0:
                # Coefficient non transmis : remplacé par sa moyenne
                coefficients[:, i, j] = moyenne
            else:
                niveaux = 2 ** bits
                pas = pas_codes[position] / 2 ** FRACTION_PAS

                debut = curseurs[bits]
                codes = np.asarray(I_encoded[bits])[debut:debut + nb_blocs]
                curseurs[bits] = debut + nb_blocs

                # Niveau de reconstruction : milieu de l'intervalle de décision
                coefficients[:, i, j] = moyenne + (codes - niveaux // 2 + 0.5) * pas

            position += 1

    T = matrice_dct(N)
    blocs = idct_2d_blocs(coefficients, T)
    image = recomposer_depuis_blocs(blocs, lignes, colonnes, N)

    return np.clip(image, 0, 255)
