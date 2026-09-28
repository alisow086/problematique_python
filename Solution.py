# ==========================================================================
#
# S7 Codage de l'information APP2 - Solution
#
# Split into two # %% cells, encoding and decoding, runnable separately. No
# data reaches the decoder except through Data, which stands in for the
# physical layer.
#
# %%
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from convert import convert
from reduce import reduce
from transmit import transmit
from computePSNR import computePSNR
from QV_encode import QV_encode, BITS_DICTIONNAIRE
from QV_decode import QV_decode
from DPCM_encode import DPCM_encode, BITS_METADATA
from DPCM_decode import DPCM_decode
from DCT_encode import DCT_encode, BITS_METADATA_DCT
from DCT_decode import DCT_decode


def imshow(numero, image, titre=""):
    """MATLAB's figure(n); imshow(image/255), greyscale aware."""
    affichage = np.clip(np.asarray(image, dtype=float) / 255.0, 0.0, 1.0)
    plt.figure(numero)
    if affichage.ndim == 2:
        plt.imshow(affichage, cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(affichage)
    plt.title(titre)
    plt.axis("off")


# ==========================================================================
#
# SÉLECTION DES PARAMÈTRES
#
# ==========================================================================
# 1 = vector quantization (QV)          4 = discrete cosine transform (DCT)
# 2 = differential quantization (DPCM)  5 = block truncation coding (BTC)
# 3 = scalar quantization (QS)          6 = adaptive quantization (QA)
Choix = 4

# QV (Choix == 1)
BLOC_L, BLOC_C = 1, 2           # block shape: horizontal pairs
K_QV = 512                      # codebook size
BITS_INDICE = int(math.log2(K_QV))   # must differ from BITS_DICTIONNAIRE,
                                     # or both would share one cell

# DPCM (Choix == 2)
NB_BITS = 4                     # bits per prediction error; 5 would give
                                # 5.000183 bits/pixel with the metadata
DENSITE = "laplacienne"         # assumed density of the prediction errors

# DCT (Choix == 4)
TAILLE_BLOC = 8                 # 8x8 blocks, like JPEG
DEBIT_CIBLE = 4.8               # mean bits per coefficient; 5.0 would give
                                # 5.04 bits/pixel with the metadata

I_source = np.asarray(Image.open("lenna.bmp"), dtype=float)
imshow(1, I_source, "Image source")
tic = time.perf_counter()

# ==========================================================================
#
# CONVERSION DE FORMAT DE CODAGE DES COULEURS
#
# ==========================================================================
I_source = convert(I_source)

# ==========================================================================
#
# RÉDUCTION DE DIMENSIONS
#
# ==========================================================================
LIGNES = 256
COLONNES = 256
I_reduced = reduce(I_source, LIGNES, COLONNES)

# ==========================================================================
#
# CODAGE
#
# ==========================================================================
# ----------------------------------------------
# CODEUR - QV
# ----------------------------------------------
if Choix == 1:
    ArgumentX = (BLOC_L, BLOC_C, K_QV)
    I_encoded, I_metadata = QV_encode(I_reduced, ArgumentX)

# ----------------------------------------------
# CODEUR - DPCM
# ----------------------------------------------
if Choix == 2:
    ArgumentX = (NB_BITS, DENSITE)
    I_encoded, I_metadata = DPCM_encode(I_reduced, ArgumentX)

# ----------------------------------------------
# CODEUR - QS
# ----------------------------------------------
if Choix == 3:
    pass  # À FAIRE : Au choix.

# ----------------------------------------------
# CODEUR - DCT
# ----------------------------------------------
if Choix == 4:
    ArgumentX = (TAILLE_BLOC, DEBIT_CIBLE)
    I_encoded, I_metadata = DCT_encode(I_reduced, ArgumentX)

# ----------------------------------------------
# CODEUR - BTC
# ----------------------------------------------
if Choix == 5:
    pass  # À FAIRE : Au choix.

# ----------------------------------------------
# CODEUR - QA
# ----------------------------------------------
if Choix == 6:
    pass  # À FAIRE : Au choix.

# ==========================================================================
#
# INTERFACE AVEC LA COUCHE PHYSIQUE
#
# ==========================================================================
# Cell N carries the values coded on N bits, so they must all fit in
# [0, 2^N - 1]. Anything the decoder needs has to travel here, metadata
# included, and it counts towards the bit budget.
Data = {}
if Choix == 1:
    Data[BITS_INDICE] = I_encoded
    Data[BITS_DICTIONNAIRE] = I_metadata
if Choix == 2:
    Data[NB_BITS] = I_encoded
    Data[BITS_METADATA] = I_metadata
if Choix == 4:
    # DCT coefficients do not share a common width: one cell per allocation
    for bits, valeurs in I_encoded.items():
        Data[bits] = valeurs
    Data[BITS_METADATA_DCT] = I_metadata

Budget = transmit(Data)
if Budget < 0:
    print(f"Erreur : Une donnée dépasse la gamme dynamique à la cellule {-Budget}.")

# %%
# ==========================================================================
#
# RECOMPOSITION
#
# ==========================================================================
if Choix == 1:
    I_encoded_Rx = Data[BITS_INDICE]
    I_metadata_Rx = Data[BITS_DICTIONNAIRE]
if Choix == 2:
    I_encoded_Rx = Data[NB_BITS]
    I_metadata_Rx = Data[BITS_METADATA]
if Choix == 4:
    I_encoded_Rx = {bits: Data[bits] for bits in Data if bits != BITS_METADATA_DCT}
    I_metadata_Rx = Data[BITS_METADATA_DCT]

# ==========================================================================
#
# DÉCODAGE
#
# ==========================================================================
# ----------------------------------------------
# DÉCODEUR - QV
# ----------------------------------------------
if Choix == 1:
    ArgumentY = (BLOC_L, BLOC_C)
    I_decoded = QV_decode(I_encoded_Rx, I_metadata_Rx, ArgumentY)

# ----------------------------------------------
# DÉCODEUR - DPCM
# ----------------------------------------------
if Choix == 2:
    ArgumentY = NB_BITS
    I_decoded = DPCM_decode(I_encoded_Rx, I_metadata_Rx, ArgumentY)

# ----------------------------------------------
# DÉCODEUR - QS
# ----------------------------------------------
if Choix == 3:
    pass  # À FAIRE : Au choix.

# ----------------------------------------------
# DÉCODEUR - DCT
# ----------------------------------------------
if Choix == 4:
    ArgumentY = (TAILLE_BLOC, LIGNES, COLONNES)
    I_decoded = DCT_decode(I_encoded_Rx, I_metadata_Rx, ArgumentY)

# ----------------------------------------------
# DÉCODEUR - BTC
# ----------------------------------------------
if Choix == 5:
    pass  # À FAIRE : Au choix.

# ----------------------------------------------
# DÉCODEUR - QA
# ----------------------------------------------
if Choix == 6:
    pass  # À FAIRE : Au choix.

imshow(2, I_decoded, "Image décodée")

# ==========================================================================
#
# CALCUL DE LA PERFORMANCE
#
# ==========================================================================
Time = time.perf_counter() - tic
if Budget > 0:
    PSNR = computePSNR(I_reduced, I_decoded)
    Rate = Budget / np.asarray(I_decoded).size
    print("********* Résultats *********")
    print(f"Temps écoulé: {Time:.2f} s")
    print(f"PSNR: {PSNR:.2f} dB")
    print(f"Rate: {Rate:.2f} bits/pixel")
    print("*****************************")

plt.show()
