import numpy as np

# Transform and metadata layout are shared with the encoder: both must walk the
# coefficients in exactly the same order.
from DCT_encode import (matrice_dct, recomposer_depuis_blocs,
                        FRACTION_PAS, OFFSET_MOYENNE)


def idct_2d_blocs(coefficients, T):
    """Inverse 2D DCT of every block."""
    blocs = np.zeros(coefficients.shape)
    for b in range(len(coefficients)):
        blocs[b] = T.T @ coefficients[b] @ T
    return blocs


def DCT_decode(I_encoded, I_metadata, ArgumentY):
    """DCT decoder. ArgumentY = (block size, rows, columns)."""
    N, lignes, colonnes = ArgumentY
    meta = np.asarray(I_metadata, dtype=int)

    B = meta[:N * N].reshape(N, N)
    pas_codes = meta[N * N:2 * N * N]
    m_codes = meta[2 * N * N:3 * N * N]

    nb_blocs = (lignes // N) * (colonnes // N)
    coefficients = np.zeros((nb_blocs, N, N))

    # Several coefficients can share a bit width, hence a cell: one read
    # cursor per cell.
    curseurs = {}
    for bits in I_encoded:
        curseurs[bits] = 0

    position = 0
    for i in range(N):
        for j in range(N):
            moyenne = m_codes[position] - OFFSET_MOYENNE
            bits = int(B[i, j])

            if bits == 0:
                # Not transmitted: replaced by its mean
                coefficients[:, i, j] = moyenne
            else:
                niveaux = 2 ** bits
                pas = pas_codes[position] / 2 ** FRACTION_PAS

                debut = curseurs[bits]
                codes = np.asarray(I_encoded[bits])[debut:debut + nb_blocs]
                curseurs[bits] = debut + nb_blocs

                coefficients[:, i, j] = moyenne + (codes - niveaux // 2 + 0.5) * pas

            position += 1

    T = matrice_dct(N)
    blocs = idct_2d_blocs(coefficients, T)
    image = recomposer_depuis_blocs(blocs, lignes, colonnes, N)

    return np.clip(image, 0, 255)
