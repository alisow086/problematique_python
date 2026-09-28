import numpy as np

# Block layout is shared with the encoder: both must walk the image in exactly
# the same order.
from QV_encode import recomposer_image


def QV_decode(I_encoded, I_metadata, ArgumentY):
    """Vector quantization decoder. ArgumentY = (bloc_l, bloc_c).

    A plain table lookup, hence the asymmetry typical of VQ: heavy encoder,
    almost free decoder.
    """
    bloc_l, bloc_c = ArgumentY
    indices = np.asarray(I_encoded, dtype=int)
    dictionnaire = np.asarray(I_metadata, dtype=float)

    nb_blocs_l, nb_blocs_c = indices.shape
    vecteurs = dictionnaire[indices.reshape(-1)]

    return recomposer_image(vecteurs, nb_blocs_l, nb_blocs_c, bloc_l, bloc_c)
