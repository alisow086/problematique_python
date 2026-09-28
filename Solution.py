# ==========================================================================
#
# S7 Codage de l'information APP2 - Solution
#
# La solution est divisée en 2 parties, le codage et le décodage. Ces deux
# sections sont délimitées par le symbole # %% et peuvent être démarrées
# séparément (PyCharm/VS Code : "Run Cell"). Aucune donnée ne peut passer
# directement du codage au décodage, i.e. sans passer par la variable Data
# qui simule la couche physique.
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
from QV_encode import QV_encode
from QV_decode import QV_decode
from DPCM_encode import DPCM_encode, BITS_METADATA
from DPCM_decode import DPCM_decode
from DCT_encode import DCT_encode, BITS_METADATA_DCT
from DCT_decode import DCT_decode


def imshow(numero, image, titre=""):
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
# Choix de la quantification
# 1 = Technique de quantification vectorielle (QV)
# 2 = Technique de quantification différentielle (DPCM)
# 3 = Technique de quantification scalaire (QS)
# 4 = Technique de quantification par transformée en cosinus discrète (DCT)
# 5 = Technique de quantification par troncature de blocs (BTC)
# 6 = Technique de quantification adaptative (QA)
Choix = 4
# Paramètres du codec QV (Choix == 1)
BLOC_L, BLOC_C = 1, 2           # forme des blocs : paires horizontales
K_QV = 512                      # taille du dictionnaire
# Nombre de bits par indice. Doit différer de 8 (la cellule du dictionnaire),
# sinon indices et dictionnaire doivent être concaténés dans la même cellule.
BITS_INDICE = int(math.log2(K_QV))

# Paramètres du codec DPCM (Choix == 2)
NB_BITS = 4                     # bits par erreur de prédiction
                                # 5 bits donnerait 5,000183 bits/pixel avec la
                                # métadonnée, au-dessus de la limite de la spec
DENSITE = "laplacienne"         # densité supposée des erreurs de prédiction

# Paramètres du codec DCT (Choix == 4)
TAILLE_BLOC = 8                 # blocs 8x8, comme JPEG
DEBIT_CIBLE = 5.0               # bits moyens par coefficient
# Charge l'image source (équivalent de im2double(imread(...))*255)
I_source: object = np.asarray(Image.open("lenna.bmp"), dtype=float)
# Affiche l'image source
imshow(1, I_source, "Image source")
# Début du chronomètre
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
# Dimensions désirées
LIGNES = 256
COLONNES = 256
# Appelle la fonction d'interpolation
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
    # Paramètres d'entrée
    ArgumentX = (BLOC_L, BLOC_C, K_QV)
    # Appelle la fonction de codage
    I_encoded, I_metadata = QV_encode(I_reduced, ArgumentX)

# ----------------------------------------------
# CODEUR - DPCM
# ----------------------------------------------
if Choix == 2:
    # Paramètres d'entrée
    ArgumentX = (NB_BITS, DENSITE)
    # Appelle la fonction de codage
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
    # Paramètres d'entrée
    ArgumentX = (TAILLE_BLOC, DEBIT_CIBLE)
    # Appelle la fonction de codage
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
# La fonction transmit envoie les données à transmettre sous un format
# compris par la couche physique. Elle prend en entrée un dictionnaire où la
# cellule N doit contenir les données qui seront codées sur N bits.
# Convention :
# Les éléments de la cellule N doivent être entre 0 et 2^N-1.
# À FAIRE : Remplir le dictionnaire Data à partir de I_encoded et
# I_metadata en respectant la convention de la couche physique.
Data = {}
if Choix == 1:
    # Les indices du dictionnaire tiennent sur BITS_INDICE bits
    Data[BITS_INDICE] = I_encoded
    # Le dictionnaire : des intensités, donc la cellule 8 bits
    Data[8] = I_metadata
if Choix == 2:
    # Les indices quantifiés tiennent par construction sur NB_BITS bits
    Data[NB_BITS] = I_encoded
    # Le pas de quantification en virgule fixe, un entier par canal
    Data[BITS_METADATA] = I_metadata
if Choix == 4:
    # Les coefficients DCT n'ont pas tous la même largeur : une cellule par
    # nombre de bits alloué, ce que la couche physique gère nativement.
    for bits, valeurs in I_encoded.items():
        Data[bits] = valeurs
    # Les trois tables : allocation, pas et moyenne par coefficient
    Data[BITS_METADATA_DCT] = I_metadata
# Appelle de la fonction de transmission
Budget = transmit(Data)
# Si une erreur a été détectée par la fonction d'interface
if Budget < 0:
    # Affichage de l'erreur
    print(f"Erreur : Une donnée dépasse la gamme dynamique à la cellule {-Budget}.")

# %%
# ==========================================================================
#
# RECOMPOSITION
#
# ==========================================================================
if Choix == 1:
    I_encoded_Rx = Data[BITS_INDICE]
    I_metadata_Rx = Data[8]
if Choix == 2:
    I_encoded_Rx = Data[NB_BITS]
    I_metadata_Rx = Data[BITS_METADATA]
if Choix == 4:
    # Toutes les cellules sauf celle des métadonnées portent des coefficients
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
    # Paramètres d'entrée
    ArgumentY = (BLOC_L, BLOC_C)
    # Appelle la fonction de décodage
    I_decoded = QV_decode(I_encoded_Rx, I_metadata_Rx, ArgumentY)

# ----------------------------------------------
# DÉCODEUR - DPCM
# ----------------------------------------------
if Choix == 2:
    # Paramètres d'entrée
    ArgumentY = NB_BITS
    # Appelle la fonction de décodage
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
    # Paramètres d'entrée
    ArgumentY = (TAILLE_BLOC, LIGNES, COLONNES)
    # Appelle la fonction de décodage
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

# Affiche l'image quantifiée
imshow(2, I_decoded, "Image décodée")

# ==========================================================================
#
# CALCUL DE LA PERFORMANCE
#
# ==========================================================================
# Fin du chronomètre
Time = time.perf_counter() - tic
# Si aucune erreur n'a été détectée
if Budget > 0:
    # Calcul du PSNR
    PSNR = computePSNR(I_reduced, I_decoded)
    # Calcul du débit
    Rate = Budget / np.asarray(I_decoded).size
    # Affichage des performances
    print("********* Résultats *********")
    print(f"Temps écoulé: {Time:.2f} s")
    print(f"PSNR: {PSNR:.2f} dB")
    print(f"Rate: {Rate:.2f} bits/pixel")
    print("*****************************")

plt.show()
