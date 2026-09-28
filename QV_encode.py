import numpy as np

# Le dictionnaire est transmis en métadonnée : ses composantes sont des
# intensités, donc la cellule 8 bits de la couche physique.
BITS_DICTIONNAIRE = 8


def decouper_en_vecteurs(image, bloc_l, bloc_c):
    """Découpe l'image en blocs et aplatit chaque bloc en un vecteur."""
    hauteur, largeur = image.shape
    blocs = image.reshape(hauteur // bloc_l, bloc_l, largeur // bloc_c, bloc_c)
    return blocs.transpose(0, 2, 1, 3).reshape(-1, bloc_l * bloc_c)


def recomposer_image(vecteurs, nb_blocs_l, nb_blocs_c, bloc_l, bloc_c):
    """Opération inverse de decouper_en_vecteurs."""
    blocs = vecteurs.reshape(nb_blocs_l, nb_blocs_c, bloc_l, bloc_c)
    return blocs.transpose(0, 2, 1, 3).reshape(nb_blocs_l * bloc_l, nb_blocs_c * bloc_c)


def plus_proches_voisins(X, C, taille_tranche=4096):
    """Indice du centroïde euclidien le plus proche pour chaque vecteur de X.

    Développe ||x - c||^2 = ||x||^2 - 2*x.c + ||c||^2. Le terme ||x||^2 est
    constant pour un x donné : il ne change pas l'argmin, on l'omet. La racine
    carrée est également inutile, elle est monotone croissante.
    Le découpage en tranches évite d'allouer une matrice N x K d'un bloc.
    """
    normes_C = (C ** 2).sum(axis=1)
    indices = np.zeros(len(X), dtype=int)
    for debut in range(0, len(X), taille_tranche):
        tranche = X[debut:debut + taille_tranche]
        distances = normes_C - 2.0 * (tranche @ C.T)
        indices[debut:debut + taille_tranche] = distances.argmin(axis=1)
    return indices


def recalculer_centroides(X, indices, C):
    """Règle du centroïde : chaque vecteur du dictionnaire devient la moyenne
    des vecteurs d'apprentissage qui lui sont assignés."""
    K, L = C.shape
    compte = np.bincount(indices, minlength=K)

    # Somme par cellule, dimension par dimension (bien plus rapide qu'un
    # masque booléen par centroïde)
    sommes = np.zeros((K, L))
    for d in range(L):
        sommes[:, d] = np.bincount(indices, weights=X[:, d], minlength=K)

    nouveau_C = C.copy()
    non_vides = compte > 0
    nouveau_C[non_vides] = sommes[non_vides] / compte[non_vides, None]

    # Cellules vides : les redéployer en dédoublant la cellule la plus peuplée,
    # sinon leur moyenne est un 0/0 qui propage des NaN dans tout le dictionnaire.
    for i in np.flatnonzero(~non_vides):
        plus_peuplee = compte.argmax()
        nouveau_C[i] = nouveau_C[plus_peuplee] * 1.01 + 0.5

    return nouveau_C


def lbg(X, K, iterations_max=30, seuil=1e-3):
    """Construit un dictionnaire de K vecteurs par l'algorithme LBG.

    Initialisation par dédoublement (splitting) : on part d'un centroïde
    unique, la moyenne globale, et on double jusqu'à atteindre K.
    """
    C = X.mean(axis=0, keepdims=True)

    while True:
        # Phase itérative de Lloyd sur le dictionnaire courant
        distorsion_precedente = np.inf
        for _ in range(iterations_max):
            indices = plus_proches_voisins(X, C)
            C = recalculer_centroides(X, indices, C)
            distorsion = ((X - C[indices]) ** 2).mean()
            # Arrêt sur l'amélioration relative de la distorsion
            if distorsion <= 0 or (distorsion_precedente - distorsion) / distorsion < seuil:
                break
            distorsion_precedente = distorsion

        if len(C) >= K:
            return C[:K]

        # Dédoublement : chaque centroïde donne deux versions perturbées
        C = np.vstack([C * 0.99, C * 1.01])[:K]


def QV_encode(I_reduced, ArgumentX):
    """Codeur par quantification vectorielle. ArgumentX = (bloc_l, bloc_c, K).

    Retourne I_encoded (indices dans [0, K-1]) et I_metadata (le dictionnaire,
    composantes entières dans [0, 255]).
    """
    bloc_l, bloc_c, K = ArgumentX
    I_S = np.asarray(I_reduced, dtype=float)

    # Ensemble d'apprentissage : l'image elle-même, découpée en blocs
    X = decouper_en_vecteurs(I_S, bloc_l, bloc_c)
    C = lbg(X, K)

    # Le dictionnaire voyage en entiers 8 bits. On l'arrondit AVANT la
    # classification finale, pour que les indices transmis désignent bien
    # les vecteurs dont le décodeur disposera.
    dictionnaire = np.clip(np.round(C), 0, 255)
    indices = plus_proches_voisins(X, dictionnaire)

    nb_blocs_l = I_S.shape[0] // bloc_l
    nb_blocs_c = I_S.shape[1] // bloc_c
    I_encoded = indices.reshape(nb_blocs_l, nb_blocs_c)

    return I_encoded, dictionnaire.astype(int)
