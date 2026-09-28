import math

import numpy as np

# Tableau 9.3 : pas de quantification normalisé (Δ/σ) et SNR théorique (dB),
# pour un quantificateur uniforme optimal. Indexé par [densité][taille d'alphabet].
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

# Le pas de quantification est une métadonnée que la couche physique n'accepte
# qu'en entier. Il est transmis en virgule fixe : pas_code = Δ * 2^FRACTION_PAS,
# dans une cellule de BITS_METADATA bits.
FRACTION_PAS = 4
BITS_METADATA = 12

# Facteurs d'élargissement du pas essayés par le codeur (voir DPCM_encode).
# L'optimum dépend de l'image : mesuré de 0,9 sur mandrill à 1,5 sur cman.
FACTEURS_PAS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]


def median3(a, b, c):
    """Médiane de trois valeurs, sans passer par np.median (beaucoup plus rapide)."""
    return max(min(a, b), min(max(a, b), c))


def predicteur_median(I_D, l, c):
    """Prédicteur du procédural : médiane de trois moyennes de voisins.

    Conservé pour référence. Le codec utilise predicteur_MED, qui traite
    beaucoup mieux les contours francs.
    """
    if l == 0 and c == 0:
        return 0.0
    elif l == 0:
        return I_D[0, c - 1]
    elif c == 0:
        return I_D[l - 1, 0]
    else:
        # Formules basées sur la matrice reconstruite I_D
        gauche = I_D[l, c - 1]
        haut = I_D[l - 1, c]
        haut_gauche = I_D[l - 1, c - 1]
        return median3((gauche + haut) / 2.0,
                       (haut_gauche + haut) / 2.0,
                       (haut_gauche + gauche) / 2.0)


def predicteur_MED(I_D, l, c):
    """Prédicteur MED (Median Edge Detector), celui de JPEG-LS.

    Contrairement à une moyenne de voisins, il détecte les contours : sur une
    transition franche, une moyenne prédirait une valeur intermédiaire qui
    n'existe nulle part, d'où une erreur maximale là où elle coûte le plus cher.
    """
    if l == 0 and c == 0:
        return 0.0
    elif l == 0:
        return I_D[0, c - 1]
    elif c == 0:
        return I_D[l - 1, 0]

    ouest = I_D[l, c - 1]
    nord = I_D[l - 1, c]
    nord_ouest = I_D[l - 1, c - 1]

    # Si le nord-ouest est hors de l'intervalle [ouest, nord], un contour
    # passe entre les deux voisins : on prend celui du bon côté.
    if nord_ouest >= max(ouest, nord):
        return min(ouest, nord)
    if nord_ouest <= min(ouest, nord):
        return max(ouest, nord)

    # Zone lisse : extrapolation planaire, qui prolonge le gradient local
    # au lieu de l'aplatir comme le ferait une moyenne.
    return ouest + nord - nord_ouest


def estimer_sigma_erreur(canal):
    """Écart-type des erreurs de prédiction, estimé en boucle ouverte sur la source.

    Version vectorisée : la prédiction se fait ici sur le canal source et non sur
    l'image reconstruite, donc tous les pixels sont indépendants.
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

    # Bords : mêmes règles que le prédicteur séquentiel
    I_P[0, :] = gauche[0, :]     # première ligne  -> pixel de gauche
    I_P[:, 0] = haut[:, 0]       # première colonne -> pixel du haut
    I_P[0, 0] = 0.0

    return float((canal - I_P).std())


def encoder_canal(canal, nb_bits, pas):
    """Boucle DPCM fermée sur un seul canal.

    Retourne les indices quantifiés et la reconstruction du décodeur local.
    Cette reconstruction est exactement ce que produira le vrai décodeur :
    elle permet au codeur de mesurer sa propre distorsion avant de transmettre.
    """
    hauteur, largeur = canal.shape
    nb_niveaux = 2 ** nb_bits
    k_min = -nb_niveaux // 2
    k_max = nb_niveaux // 2
    # Niveaux de reconstruction de l'erreur (quantificateur centré en 0)
    reconstructions = (np.arange(k_min, k_max) + 0.5) * pas

    I_E = np.zeros((hauteur, largeur), dtype=int)
    I_D = np.zeros((hauteur, largeur), dtype=float)

    for l in range(hauteur):
        for c in range(largeur):
            # a. Prédiction depuis I_D (les voisins déjà reconstruits)
            Ip = predicteur_MED(I_D, l, c)
            # b. Erreur de prédiction
            e = canal[l, c] - Ip
            # c. Quantification de l'erreur
            k_pixel = math.floor(e / pas)
            k_pixel = max(k_min, min(k_pixel, k_max - 1))
            index_code = k_pixel - k_min
            I_E[l, c] = index_code
            # d. Reconstruction immédiate du pixel, identique à celle du décodeur
            I_D[l, c] = Ip + reconstructions[index_code]

    return I_E, I_D


def DPCM_encode(I_reduced, ArgumentX):
    """Codeur DPCM. ArgumentX = (nb_bits, densite).

    Retourne I_encoded (indices dans [0, 2^nb_bits - 1]) et I_metadata
    (le pas de quantification en virgule fixe, un par canal).
    """
    nb_bits, densite = ArgumentX
    if 2 ** nb_bits not in TABLEAU_9_3[densite]:
        raise ValueError(
            f"nb_bits = {nb_bits} donne {2 ** nb_bits} niveaux, absent du tableau 9.3 "
            f"(tailles disponibles : {sorted(TABLEAU_9_3[densite])}). Utilisez nb_bits <= 5."
        )
    I_S = np.asarray(I_reduced, dtype=float)

    # Traite indifféremment une image monochrome (H, W) ou couleur (H, W, C)
    canaux = [I_S] if I_S.ndim == 2 else [I_S[:, :, i] for i in range(I_S.shape[2])]

    pas_normalise, _ = TABLEAU_9_3[densite][2 ** nb_bits]
    I_encoded_canaux = []
    I_metadata = []

    for canal in canaux:
        sigma_e = estimer_sigma_erreur(canal)

        # Le pas du tableau 9.3 est optimal pour une source EXACTEMENT
        # laplacienne. Les erreurs de prédiction réelles ont des queues plus
        # lourdes, donc ce pas est trop fin et sature sur les contours.
        # Le codeur possédant un décodeur local, il peut essayer plusieurs pas,
        # mesurer la distorsion réelle de chacun et garder le meilleur.
        # Le facteur n'est jamais transmis : il est absorbé dans la valeur du
        # pas, qui voyage déjà en métadonnée. Coût en débit : nul.
        meilleur = (np.inf, None, None)
        for facteur in FACTEURS_PAS:
            # Le pas est arrondi AVANT d'encoder : le codeur doit utiliser
            # exactement la valeur que le décodeur recevra, sinon les deux
            # boucles divergent et l'erreur se propage.
            pas_code = int(round(pas_normalise * sigma_e * facteur * 2 ** FRACTION_PAS))
            pas_code = min(max(pas_code, 1), 2 ** BITS_METADATA - 1)

            I_E, I_D = encoder_canal(canal, nb_bits, pas_code / 2 ** FRACTION_PAS)
            # Distorsion telle que le décodeur la produira (clipping inclus)
            eqm = ((canal - np.clip(I_D, 0, 255)) ** 2).mean()
            if eqm < meilleur[0]:
                meilleur = (eqm, pas_code, I_E)

        I_encoded_canaux.append(meilleur[2])
        I_metadata.append(meilleur[1])

    I_encoded = I_encoded_canaux[0] if I_S.ndim == 2 else np.stack(I_encoded_canaux, axis=2)
    return I_encoded, np.array(I_metadata, dtype=int)
