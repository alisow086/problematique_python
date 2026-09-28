import numpy as np

# ITU-R BT.601 luminance weights: the eye is far more sensitive to green
# than to blue, hence the uneven coefficients.
POIDS_LUMINANCE = np.array([0.299, 0.587, 0.114])


def convert(I_source):
    """Reduce colour depth: 24-bit RGB -> 8-bit greyscale.

    Accepts an (H, W, 3) RGB image or an (H, W) greyscale one, always returns
    a 2D array.
    """
    I_source = np.asarray(I_source, dtype=float)

    if I_source.ndim == 2:
        return I_source

    # [:, :, :3] drops any alpha channel
    return I_source[:, :, :3] @ POIDS_LUMINANCE
