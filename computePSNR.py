import numpy as np


def computePSNR(I_reduced, I_decoded):
    I_reduced = np.asarray(I_reduced, dtype=float)
    I_decoded = np.asarray(I_decoded, dtype=float)
    # Différence de l'image synthétisée et de l'image source
    image_MSE = I_decoded - I_reduced
    # Réarrangement sur une ligne
    image_MSE = image_MSE.reshape(-1)
    # Calcul de l'erreur quadratique moyenne
    MSE = np.sum(image_MSE ** 2) / image_MSE.size
    # Images identiques : PSNR infini (MATLAB retourne Inf de la même façon)
    if MSE == 0:
        return float("inf")
    # Calcul du PSNR
    return 10 * np.log10(255 ** 2 / MSE)
