import numpy as np

# Predictor and metadata format are shared with the encoder: both loops must be
# rigorously identical.
from DPCM_encode import FRACTION_PAS, predicteur_MED


def decoder_canal(I_E, nb_bits, pas):
    """DPCM reconstruction loop on one channel."""
    hauteur, largeur = I_E.shape
    nb_niveaux = 2 ** nb_bits
    k_min = -nb_niveaux // 2
    k_max = nb_niveaux // 2
    reconstructions = (np.arange(k_min, k_max) + 0.5) * pas

    I_D = np.zeros((hauteur, largeur), dtype=float)

    for l in range(hauteur):
        for c in range(largeur):
            Ip = predicteur_MED(I_D, l, c)
            I_D[l, c] = Ip + reconstructions[I_E[l, c]]

    return I_D


def DPCM_decode(I_encoded, I_metadata, ArgumentY):
    """DPCM decoder. ArgumentY = nb_bits.

    I_metadata holds the quantization step as fixed point, one per channel.
    """
    nb_bits = ArgumentY
    I_E = np.asarray(I_encoded, dtype=int)
    metadata = np.atleast_1d(np.asarray(I_metadata, dtype=int))

    canaux = [I_E] if I_E.ndim == 2 else [I_E[:, :, i] for i in range(I_E.shape[2])]

    I_decoded_canaux = [
        decoder_canal(canal, nb_bits, metadata[i] / 2 ** FRACTION_PAS)
        for i, canal in enumerate(canaux)
    ]

    I_decoded = I_decoded_canaux[0] if I_E.ndim == 2 else np.stack(I_decoded_canaux, axis=2)

    # Clip only after the loop: during it, encoder and decoder must handle the
    # exact same unbounded values.
    return np.clip(I_decoded, 0, 255)
