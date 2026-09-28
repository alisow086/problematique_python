import math

import numpy as np


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

# Densité supposée des coefficients DCT. Passer à "uniforme" pour reproduire
# l'hypothèse du procédural et constater le plafonnement du PSNR.
DENSITE_COEFFICIENTS = "laplacienne"

# Les trois tables de métadonnées (allocation, pas, moyenne) voyagent dans une
# seule cellule de 14 bits. 14 ne peut pas entrer en collision avec les cellules
# des coefficients, qui sont limitées à BITS_MAX.
BITS_METADATA_DCT = 14
FRACTION_PAS = 4        # pas transmis en virgule fixe : pas * 2^4
OFFSET_MOYENNE = 2048   # les moyennes des coefficients AC sont signées
BITS_MAX = 8            # plafond d'allocation par coefficient


def matrice_dct(N):

    T = np.zeros((N, N))
    for k in range(N):
        alpha = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
        for n in range(N):
            T[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
    return T


def pas_normalise(bits, densite=DENSITE_COEFFICIENTS):
    """Pas normalisé du tableau 9.3 pour 2^bits niveaux.

    Le tableau s'arrête à 32 niveaux (5 bits). Au-delà, on extrapole en
    divisant le pas par deux à chaque bit ajouté : la plage reste alors celle
    de 32 niveaux, déjà assez large pour ne plus rien écrêter.
    """
    table = TABLEAU_9_3[densite]
    niveaux = 2 ** bits
    if niveaux in table:
        return table[niveaux]
    return table[32] / 2 ** (bits - 5)


def decouper_en_blocs(image, N):
    """Découpe l'image en blocs N x N, empilés sur la première dimension."""
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
    """Opération inverse de decouper_en_blocs."""
    nb_blocs_c = largeur // N
    image = np.zeros((hauteur, largeur))
    for indice in range(len(blocs)):
        bl = indice // nb_blocs_c
        bc = indice % nb_blocs_c
        image[bl * N:(bl + 1) * N, bc * N:(bc + 1) * N] = blocs[indice]
    return image


def dct_2d_blocs(blocs, T):
    """DCT 2D de chaque bloc : C = T @ bloc @ T.T."""
    coefficients = np.zeros(blocs.shape)
    for b in range(len(blocs)):
        coefficients[b] = T @ blocs[b] @ T.T
    return coefficients


def statistiques_coefficients(coefficients, N):
    """Moyenne et variance de chaque position de coefficient, à travers les blocs.

    C'est la statistique centrale de la méthode : le coefficient continu (0,0)
    a une variance énorme, les hautes fréquences sont quasi nulles partout.
    """
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
    """Nombre total de bits alloués."""
    total = 0
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            total += int(B[i, j])
    return total


def gain_marginal(variance, bits):
    """Réduction de distorsion qu'apporterait un bit supplémentaire.

    La distorsion d'un quantificateur à B bits décroît comme var * 4^-B, donc
    le gain du bit suivant vaut var * 4^-B. C'est ce gain, et non la variance
    seule, qui doit arbitrer la redistribution : un coefficient très variant
    mais déjà riche en bits rapporte moins qu'un coefficient moyen encore pauvre.
    """
    return variance * (4.0 ** -bits)


def allouer_bits(variances, debit_cible, bits_max=BITS_MAX):
    """Allocation des bits par coefficient (formule du procédural 2, problème 3).

    B_k = B/(L*C) + 1/2 * log2( var_k / moyenne_géométrique(var) )

    Un coefficient dont la variance dépasse la moyenne géométrique reçoit plus
    que sa part ; les hautes fréquences, presque nulles partout, tombent à zéro
    et ne sont pas transmises du tout. C'est de là que vient la compression.
    """
    N = variances.shape[0]

    # Moyenne géométrique = exponentielle de la moyenne des logarithmes
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

    # L'arrondi et l'écrêtage cassent le budget exact, et l'écart peut être
    # important quand beaucoup de coefficients sont écrêtés à 0 ou à bits_max.
    # On le rattrape au gain marginal : chaque bit va là où il rapporte le plus,
    # chaque bit retiré vient de là où il rapportait le moins.
    cible = int(round(debit_cible * N * N))
    total = total_bits(B)

    while total > cible:
        choix_i, choix_j, plus_faible = -1, -1, math.inf
        for i in range(N):
            for j in range(N):
                if B[i, j] > 0:
                    # Gain que procurait le dernier bit attribué
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
    """Codeur par transformée en cosinus discrète.

    ArgumentX = (taille_bloc, debit_cible) où debit_cible est le nombre moyen
    de bits par coefficient.

    Retourne I_encoded, un dictionnaire {nombre de bits: indices quantifiés}
    (les coefficients n'ont pas tous la même largeur), et I_metadata,
    la concaténation des trois tables allocation / pas / moyenne.
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
            # La moyenne est transmise pour tous les coefficients : le décodeur
            # s'en sert telle quelle pour ceux qui ne reçoivent aucun bit.
            m_code = int(round(moyennes[i, j])) + OFFSET_MOYENNE
            m_codes[position] = min(max(m_code, 0), 2 ** BITS_METADATA_DCT - 1)

            bits = int(B[i, j])
            if bits > 0:
                niveaux = 2 ** bits
                pas = pas_normalise(bits) * math.sqrt(variances[i, j])

                # Le pas est arrondi AVANT de quantifier, pour que le codeur
                # utilise exactement la valeur que le décodeur recevra.
                pas_code = int(round(pas * 2 ** FRACTION_PAS))
                pas_codes[position] = min(max(pas_code, 1),
                                          2 ** BITS_METADATA_DCT - 1)
                pas = pas_codes[position] / 2 ** FRACTION_PAS
                moyenne = m_codes[position] - OFFSET_MOYENNE

                # Quantificateur uniforme midrise centré sur la moyenne
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
