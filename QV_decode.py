import numpy as np

# Le découpage en blocs est partagé avec le codeur : les deux doivent
# parcourir l'image dans exactement le même ordre.
from QV_encode import recomposer_image


def QV_decode(I_encoded, I_metadata, ArgumentY):
    """Décodeur par quantification vectorielle. ArgumentY = (bloc_l, bloc_c).

    Simple lecture de table : chaque indice est remplacé par son vecteur du
    dictionnaire. Aucun calcul, d'où l'asymétrie caractéristique de la QV
    (codeur lourd, décodeur quasi gratuit).
    """
    bloc_l, bloc_c = ArgumentY
    indices = np.asarray(I_encoded, dtype=int)
    dictionnaire = np.asarray(I_metadata, dtype=float)

    nb_blocs_l, nb_blocs_c = indices.shape
    vecteurs = dictionnaire[indices.reshape(-1)]

    return recomposer_image(vecteurs, nb_blocs_l, nb_blocs_c, bloc_l, bloc_c)
