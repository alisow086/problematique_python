import math

import numpy as np

# Table 9.3: (normalized step delta/sigma, theoretical SNR in dB) for an
# optimal uniform quantizer, indexed by [density][alphabet size].
TABLEAU_9_3 = {
    "uniforme": {
        2: (1.732, 6.02),  4: (0.866, 12.04),  6: (0.577, 15.58),  8: (0.433, 18.06),
        10: (0.346, 20.02), 12: (0.289, 21.60), 14: (0.247, 22.94), 16: (0.217, 24.08),
        32: (0.108, 30.10),
    },
    "gaussienne": {
        2: (1.596, 4.40),  4: (0.9957, 9.24),  6: (0.7334, 12.18), 8: (0.5860, 14.27),
        10: (0.4908, 15.90), 12: (0.4238, 17.25), 14: (0.3739, 18.37), 16: (0.3352, 19.36),
        32: (0.1881, 24.56),
    },
    "laplacienne": {
        2: (1.414, 3.00),  4: (1.0873, 7.05),  6: (0.8707, 9.56),  8: (0.7309, 11.39),
        10: (0.6334, 12.81), 12: (0.5613, 13.98), 14: (0.5055, 14.98), 16: (0.4609, 15.84),
        32: (0.2799, 20.46),
    },
}

# The physical layer only carries integers, so the step travels as fixed point:
# pas_code = delta * 2^FRACTION_PAS, in a BITS_METADATA-wide cell.
FRACTION_PAS = 4
BITS_METADATA = 12

# Step widening factors tried by the encoder. The optimum is image dependent:
# measured from 0.9 on mandrill to 1.5 on cameraman.
FACTEURS_PAS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]


def predicteur_MED(I_D, l, c):
    """MED predictor (Median Edge Detector), the one used by JPEG-LS."""
    if l == 0 and c == 0:
        return 0.0
    elif l == 0:
        return I_D[0, c - 1]
    elif c == 0:
        return I_D[l - 1, 0]

    ouest = I_D[l, c - 1]
    nord = I_D[l - 1, c]
    nord_ouest = I_D[l - 1, c - 1]

    # North-west outside [west, north] means an edge runs between the two
    # neighbours: pick the one on the right side rather than averaging, which
    # would predict a value that exists nowhere.
    if nord_ouest >= max(ouest, nord):
        return min(ouest, nord)
    if nord_ouest <= min(ouest, nord):
        return max(ouest, nord)

    # Smooth area: planar extrapolation, which extends the local gradient
    return ouest + nord - nord_ouest


def estimer_sigma_erreur(canal):
    """Standard deviation of the prediction errors, estimated open loop.

    Predicting on the source rather than on the reconstruction makes every
    pixel independent, so this pass can be vectorized.
    """
    gauche = np.zeros_like(canal)
    gauche[:, 1:] = canal[:, :-1]
    haut = np.zeros_like(canal)
    haut[1:, :] = canal[:-1, :]
    haut_gauche = np.zeros_like(canal)
    haut_gauche[1:, 1:] = canal[:-1, :-1]

    I_P = np.median(np.stack([(gauche + haut) / 2.0,
                              (haut_gauche + haut) / 2.0,
                              (haut_gauche + gauche) / 2.0]), axis=0)

    # Borders: same rules as the sequential predictor
    I_P[0, :] = gauche[0, :]
    I_P[:, 0] = haut[:, 0]
    I_P[0, 0] = 0.0

    return float((canal - I_P).std())


def encoder_canal(canal, nb_bits, pas):
    """Closed DPCM loop on one channel.

    Returns the quantized indices and the local decoder's reconstruction, which
    is exactly what the real decoder will produce. The encoder uses it to
    measure its own distortion before transmitting.
    """
    hauteur, largeur = canal.shape
    nb_niveaux = 2 ** nb_bits
    k_min = -nb_niveaux // 2
    k_max = nb_niveaux // 2
    reconstructions = (np.arange(k_min, k_max) + 0.5) * pas

    I_E = np.zeros((hauteur, largeur), dtype=int)
    I_D = np.zeros((hauteur, largeur), dtype=float)

    for l in range(hauteur):
        for c in range(largeur):
            # Predict from already reconstructed neighbours, never from the
            # source: this is what keeps encoder and decoder in step.
            Ip = predicteur_MED(I_D, l, c)
            e = canal[l, c] - Ip

            k_pixel = math.floor(e / pas)
            k_pixel = max(k_min, min(k_pixel, k_max - 1))
            index_code = k_pixel - k_min

            I_E[l, c] = index_code
            I_D[l, c] = Ip + reconstructions[index_code]

    return I_E, I_D


def DPCM_encode(I_reduced, ArgumentX):
    """DPCM encoder. ArgumentX = (nb_bits, density).

    Returns I_encoded (indices in [0, 2^nb_bits - 1]) and I_metadata (the
    quantization step as fixed point, one per channel).
    """
    nb_bits, densite = ArgumentX
    if 2 ** nb_bits not in TABLEAU_9_3[densite]:
        raise ValueError(
            f"nb_bits = {nb_bits} donne {2 ** nb_bits} niveaux, absent du tableau 9.3 "
            f"(tailles disponibles : {sorted(TABLEAU_9_3[densite])}). Utilisez nb_bits <= 5."
        )
    I_S = np.asarray(I_reduced, dtype=float)

    # Handles a monochrome (H, W) or colour (H, W, C) image alike
    canaux = [I_S] if I_S.ndim == 2 else [I_S[:, :, i] for i in range(I_S.shape[2])]

    pas_normalise, _ = TABLEAU_9_3[densite][2 ** nb_bits]
    I_encoded_canaux = []
    I_metadata = []

    for canal in canaux:
        sigma_e = estimer_sigma_erreur(canal)

        # Table 9.3 assumes an exactly Laplacian source. Real prediction errors
        # have heavier tails, so its step is too fine and saturates on edges.
        # Having a local decoder, the encoder can try several steps, measure the
        # real distortion of each and keep the best. The factor is never
        # transmitted, only the resulting step, so this costs no bits.
        meilleur = (np.inf, None, None)
        for facteur in FACTEURS_PAS:
            # Round the step before encoding, so the encoder uses exactly the
            # value the decoder will receive: otherwise both loops drift apart
            # and the error propagates.
            pas_code = int(round(pas_normalise * sigma_e * facteur * 2 ** FRACTION_PAS))
            pas_code = min(max(pas_code, 1), 2 ** BITS_METADATA - 1)

            I_E, I_D = encoder_canal(canal, nb_bits, pas_code / 2 ** FRACTION_PAS)
            eqm = ((canal - np.clip(I_D, 0, 255)) ** 2).mean()
            if eqm < meilleur[0]:
                meilleur = (eqm, pas_code, I_E)

        I_encoded_canaux.append(meilleur[2])
        I_metadata.append(meilleur[1])

    I_encoded = I_encoded_canaux[0] if I_S.ndim == 2 else np.stack(I_encoded_canaux, axis=2)
    return I_encoded, np.array(I_metadata, dtype=int)
