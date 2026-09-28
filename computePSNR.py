import numpy as np


def computePSNR(I_reduced, I_decoded):
    """Peak signal-to-noise ratio, in dB, between source and reconstruction."""
    I_reduced = np.asarray(I_reduced, dtype=float)
    I_decoded = np.asarray(I_decoded, dtype=float)

    image_MSE = (I_decoded - I_reduced).reshape(-1)
    MSE = np.sum(image_MSE ** 2) / image_MSE.size

    # Identical images: infinite PSNR, as MATLAB also returns
    if MSE == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / MSE)
