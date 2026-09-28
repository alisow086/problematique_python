import numpy as np

# The codebook holds intensities, so it travels in the 8-bit cell.
BITS_DICTIONNAIRE = 8


def decouper_en_vecteurs(image, bloc_l, bloc_c):
    """Split the image into blocks, each flattened into one vector."""
    hauteur, largeur = image.shape
    blocs = image.reshape(hauteur // bloc_l, bloc_l, largeur // bloc_c, bloc_c)
    return blocs.transpose(0, 2, 1, 3).reshape(-1, bloc_l * bloc_c)


def recomposer_image(vecteurs, nb_blocs_l, nb_blocs_c, bloc_l, bloc_c):
    """Inverse of decouper_en_vecteurs."""
    blocs = vecteurs.reshape(nb_blocs_l, nb_blocs_c, bloc_l, bloc_c)
    return blocs.transpose(0, 2, 1, 3).reshape(nb_blocs_l * bloc_l, nb_blocs_c * bloc_c)


def plus_proches_voisins(X, C, taille_tranche=4096):
    """Index of the nearest centroid (euclidean) for every vector of X.

    Expands ||x - c||^2 = ||x||^2 - 2*x.c + ||c||^2 and drops ||x||^2, constant
    for a given x: same argmin, one matrix product instead of a loop over K.
    Slicing keeps the N x K distance matrix from being allocated at once.
    """
    normes_C = (C ** 2).sum(axis=1)
    indices = np.zeros(len(X), dtype=int)
    for debut in range(0, len(X), taille_tranche):
        tranche = X[debut:debut + taille_tranche]
        distances = normes_C - 2.0 * (tranche @ C.T)
        indices[debut:debut + taille_tranche] = distances.argmin(axis=1)
    return indices


def recalculer_centroides(X, indices, C):
    """Centroid rule: each codevector becomes the mean of its assigned vectors."""
    K, L = C.shape
    compte = np.bincount(indices, minlength=K)

    # Per-cell sums, one dimension at a time: L passes instead of K
    sommes = np.zeros((K, L))
    for d in range(L):
        sommes[:, d] = np.bincount(indices, weights=X[:, d], minlength=K)

    nouveau_C = C.copy()
    non_vides = compte > 0
    nouveau_C[non_vides] = sommes[non_vides] / compte[non_vides, None]

    # An empty cell has a 0/0 mean that would spread NaN through the whole
    # codebook: redeploy it next to the most populated one instead.
    for i in np.flatnonzero(~non_vides):
        plus_peuplee = compte.argmax()
        nouveau_C[i] = nouveau_C[plus_peuplee] * 1.01 + 0.5

    return nouveau_C


def lbg(X, K, iterations_max=30, seuil=1e-3):
    """Build a K-vector codebook with the LBG algorithm.

    Initialized by splitting: start from the global mean and double until K.
    Slower than a random draw but deterministic, which matters for validation.
    """
    C = X.mean(axis=0, keepdims=True)

    while True:
        distorsion_precedente = np.inf
        for _ in range(iterations_max):
            indices = plus_proches_voisins(X, C)
            C = recalculer_centroides(X, indices, C)
            distorsion = ((X - C[indices]) ** 2).mean()
            if distorsion <= 0 or (distorsion_precedente - distorsion) / distorsion < seuil:
                break
            distorsion_precedente = distorsion

        if len(C) >= K:
            return C[:K]

        C = np.vstack([C * 0.99, C * 1.01])[:K]


def QV_encode(I_reduced, ArgumentX):
    """Vector quantization encoder. ArgumentX = (bloc_l, bloc_c, K).

    Returns I_encoded (indices in [0, K-1]) and I_metadata (the codebook, with
    integer components in [0, 255]).
    """
    bloc_l, bloc_c, K = ArgumentX
    I_S = np.asarray(I_reduced, dtype=float)

    # Training set: the image itself, split into blocks
    X = decouper_en_vecteurs(I_S, bloc_l, bloc_c)
    C = lbg(X, K)

    # Round the codebook before the final classification, so the transmitted
    # indices point at the vectors the decoder will actually hold.
    dictionnaire = np.clip(np.round(C), 0, 255)
    indices = plus_proches_voisins(X, dictionnaire)

    nb_blocs_l = I_S.shape[0] // bloc_l
    nb_blocs_c = I_S.shape[1] // bloc_c
    I_encoded = indices.reshape(nb_blocs_l, nb_blocs_c)

    return I_encoded, dictionnaire.astype(int)
