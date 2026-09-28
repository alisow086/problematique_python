import math

import numpy as np

# Table 9.3: normalized quantization step (delta / sigma) by alphabet size.
TABLEAU_9_3 = {
    "uniforme": {
        2: 1.732, 4: 0.866, 6: 0.577, 8: 0.433, 10: 0.346,
        12: 0.289, 14: 0.247, 16: 0.217, 32: 0.108,
    },
    "laplacienne": {
        2: 1.414, 4: 1.0873, 6: 0.8707, 8: 0.7309, 10: 0.6334,
        12: 0.5613, 14: 0.5055, 16: 0.4609, 32: 0.2799,
    },
}

# The uniform row keeps the quantizer range at 3.46 sigma whatever the bit
# count, so the energy it clips never shrinks and the PSNR plateaus near 29 dB.
# DCT coefficients are peaked with heavy tails; the Laplacian row widens the
# range as bits are added and is worth over 12 dB here.
DENSITE_COEFFICIENTS = "laplacienne"

# The three metadata tables share one 14-bit cell. 14 cannot collide with the
# coefficient cells, which are capped at BITS_MAX.
BITS_METADATA_DCT = 14
FRACTION_PAS = 4        # step sent as fixed point: step * 2^4
OFFSET_MOYENNE = 2048   # AC coefficient means are signed
BITS_MAX = 8            # per-coefficient allocation ceiling


def matrice_dct(N):
    """Orthonormal DCT-II matrix. The 2D DCT of a block is T @ block @ T.T."""
    T = np.zeros((N, N))
    for k in range(N):
        alpha = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
        for n in range(N):
            T[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
    return T


def pas_normalise(bits, densite=DENSITE_COEFFICIENTS):
    """Normalized step from table 9.3, extrapolated past its 32-level limit."""
    table = TABLEAU_9_3[densite]
    niveaux = 2 ** bits
    if niveaux in table:
        return table[niveaux]
    return table[32] / 2 ** (bits - 5)


def decouper_en_blocs(image, N):
    """Split the image into N x N blocks, stacked along the first axis."""
    hauteur, largeur = image.shape
    nb_blocs_l = hauteur // N
    nb_blocs_c = largeur // N

    blocs = np.zeros((nb_blocs_l * nb_blocs_c, N, N))
    indice = 0
    for bl in range(nb_blocs_l):
        for bc in range(nb_blocs_c):
            blocs[indice] = image[bl * N:(bl + 1) * N, bc * N:(bc + 1) * N]
            indice += 1
    return blocs


def recomposer_depuis_blocs(blocs, hauteur, largeur, N):
    """Inverse of decouper_en_blocs."""
    nb_blocs_c = largeur // N
    image = np.zeros((hauteur, largeur))
    for indice in range(len(blocs)):
        bl = indice // nb_blocs_c
        bc = indice % nb_blocs_c
        image[bl * N:(bl + 1) * N, bc * N:(bc + 1) * N] = blocs[indice]
    return image


def dct_2d_blocs(blocs, T):
    """2D DCT of every block."""
    coefficients = np.zeros(blocs.shape)
    for b in range(len(blocs)):
        coefficients[b] = T @ blocs[b] @ T.T
    return coefficients


def statistiques_coefficients(coefficients, N):
    """Mean and variance of each coefficient position, across all blocks."""
    nb_blocs = len(coefficients)
    moyennes = np.zeros((N, N))
    variances = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            total = 0.0
            for b in range(nb_blocs):
                total += coefficients[b, i, j]
            moyenne = total / nb_blocs

            ecart = 0.0
            for b in range(nb_blocs):
                ecart += (coefficients[b, i, j] - moyenne) ** 2

            moyennes[i, j] = moyenne
            variances[i, j] = ecart / nb_blocs

    return moyennes, variances


def total_bits(B):
    """Total number of allocated bits."""
    total = 0
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            total += int(B[i, j])
    return total


def gain_marginal(variance, bits):
    """Distortion reduction one more bit would buy, var * 4^-bits."""
    return variance * (4.0 ** -bits)


def allouer_bits(variances, debit_cible, bits_max=BITS_MAX):
    """Per-coefficient bit allocation (procedural 2, problem 3).

    B_k = B/(L*C) + 1/2 * log2( var_k / geometric_mean(var) )

    High frequencies are near zero everywhere, fall to zero bits and are not
    transmitted at all. That is where the compression comes from.
    """
    N = variances.shape[0]

    total_log = 0.0
    for i in range(N):
        for j in range(N):
            total_log += math.log(max(variances[i, j], 1e-12))
    moyenne_geometrique = math.exp(total_log / (N * N))

    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            variance = max(variances[i, j], 1e-12)
            bits = debit_cible + 0.5 * math.log2(variance / moyenne_geometrique)
            B[i, j] = int(min(max(round(bits), 0), bits_max))

    # Rounding and clamping break the exact budget, sometimes by a wide margin.
    # Redistribute by marginal gain, not by raw variance: the latter piles bits
    # onto a few coefficients until they saturate and starves the rest.
    cible = int(round(debit_cible * N * N))
    total = total_bits(B)

    while total > cible:
        choix_i, choix_j, plus_faible = -1, -1, math.inf
        for i in range(N):
            for j in range(N):
                if B[i, j] > 0:
                    gain = gain_marginal(variances[i, j], B[i, j] - 1)
                    if gain < plus_faible:
                        plus_faible = gain
                        choix_i, choix_j = i, j
        B[choix_i, choix_j] -= 1
        total -= 1

    while total < cible:
        choix_i, choix_j, plus_fort = -1, -1, -math.inf
        for i in range(N):
            for j in range(N):
                if B[i, j] < bits_max:
                    gain = gain_marginal(variances[i, j], B[i, j])
                    if gain > plus_fort:
                        plus_fort = gain
                        choix_i, choix_j = i, j
        B[choix_i, choix_j] += 1
        total += 1

    return B


def DCT_encode(I_reduced, ArgumentX):
    """DCT encoder. ArgumentX = (block size, mean bits per coefficient).

    Returns I_encoded as {bit width: quantized indices} since coefficients do
    not share a common width, and I_metadata as the allocation, step and mean
    tables concatenated.
    """
    N, debit_cible = ArgumentX
    I_S = np.asarray(I_reduced, dtype=float)
    T = matrice_dct(N)

    blocs = decouper_en_blocs(I_S, N)
    coefficients = dct_2d_blocs(blocs, T)
    moyennes, variances = statistiques_coefficients(coefficients, N)
    B = allouer_bits(variances, debit_cible)

    pas_codes = np.zeros(N * N, dtype=int)
    m_codes = np.zeros(N * N, dtype=int)
    cellules = {}

    position = 0
    for i in range(N):
        for j in range(N):
            # Sent for every coefficient: the decoder uses it as-is for those
            # that receive no bits.
            m_code = int(round(moyennes[i, j])) + OFFSET_MOYENNE
            m_codes[position] = min(max(m_code, 0), 2 ** BITS_METADATA_DCT - 1)

            bits = int(B[i, j])
            if bits > 0:
                niveaux = 2 ** bits
                pas = pas_normalise(bits) * math.sqrt(variances[i, j])

                # Round the step before quantizing, so the encoder uses exactly
                # the value the decoder will receive.
                pas_code = int(round(pas * 2 ** FRACTION_PAS))
                pas_codes[position] = min(max(pas_code, 1),
                                          2 ** BITS_METADATA_DCT - 1)
                pas = pas_codes[position] / 2 ** FRACTION_PAS
                moyenne = m_codes[position] - OFFSET_MOYENNE

                # Midrise uniform quantizer centred on the coefficient mean
                k = np.floor((coefficients[:, i, j] - moyenne) / pas)
                k = np.clip(k, -niveaux // 2, niveaux // 2 - 1)
                codes = (k + niveaux // 2).astype(int)

                if bits not in cellules:
                    cellules[bits] = []
                cellules[bits].append(codes)

            position += 1

    I_encoded = {bits: np.concatenate(listes) for bits, listes in cellules.items()}
    I_metadata = np.concatenate([B.reshape(-1), pas_codes, m_codes])

    return I_encoded, I_metadata
