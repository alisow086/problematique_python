import numpy as np

# Le prédicteur et le format de la métadonnée sont partagés avec le codeur :
# les deux boucles doivent être rigoureusement identiques.
from DPCM_encode import FRACTION_PAS, predicteur_MED


def decoder_canal(I_E, nb_bits, pas):
    """Boucle DPCM de reconstruction sur un seul canal."""
    hauteur, largeur = I_E.shape
    nb_niveaux = 2 ** nb_bits
    k_min = -nb_niveaux // 2
    k_max = nb_niveaux // 2
    # Mêmes niveaux de reconstruction que ceux utilisés par le codeur
    reconstructions = (np.arange(k_min, k_max) + 0.5) * pas

    I_D = np.zeros((hauteur, largeur), dtype=float)

    for l in range(hauteur):
        for c in range(largeur):
            # Prédiction depuis les pixels déjà reconstruits
            Ip = predicteur_MED(I_D, l, c)
            # Pixel = prédiction + erreur quantifiée
            I_D[l, c] = Ip + reconstructions[I_E[l, c]]

    return I_D


def DPCM_decode(I_encoded, I_metadata, ArgumentY):
    """Décodeur DPCM. ArgumentY = nb_bits.

    I_metadata contient le pas de quantification en virgule fixe, un par canal.
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

    # Une image 8 bits ne peut pas stocker de valeur hors de [0, 255].
    # Le clipping se fait APRÈS la boucle : pendant la boucle, codeur et
    # décodeur doivent manipuler exactement les mêmes valeurs non bornées.
    return np.clip(I_decoded, 0, 255)
