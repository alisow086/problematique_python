import numpy as np


def transmit(Data):
    """Interface avec la couche physique.

    Data est un dictionnaire {N: donnees} où la clé N est le nombre de bits
    sur lequel les données de la cellule seront codées (équivalent du tableau
    de cellules Data{N} de MATLAB, dont les indices commencent aussi à 1).

    Retourne le budget en bits, ou -N si la cellule N dépasse sa gamme
    dynamique [0, 2^N - 1].
    """
    # Initialisation
    budget_data = 0
    if not Data:
        return budget_data

    # Pour chaque cellule
    for i in range(1, max(Data) + 1):
        cellule = np.asarray(Data.get(i, []))
        # Une cellule vide ne transporte aucune donnée
        if cellule.size == 0:
            continue
        # Si la donnée est plus petite que la gamme dynamique
        if cellule.min() < 0:
            # Sortir un code d'erreur
            return -i
        # Si la donnée est plus grande que la gamme dynamique
        if cellule.max() > 2 ** i - 1:
            # Sortir un code d'erreur
            return -i
        # Calcul de la quantité de données dans cette cellule
        budget_data += cellule.size * i

    return budget_data
