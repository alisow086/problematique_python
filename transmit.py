import numpy as np


def transmit(Data):
    """Physical layer interface.

    Data maps a bit width N to the values that will be coded on N bits, the
    equivalent of the MATLAB cell array Data{N} (also 1-indexed).
    Returns the budget in bits, or -N if cell N exceeds [0, 2^N - 1].
    """
    budget_data = 0
    if not Data:
        return budget_data

    for i in range(1, max(Data) + 1):
        cellule = np.asarray(Data.get(i, []))
        if cellule.size == 0:
            continue
        if cellule.min() < 0:
            return -i
        if cellule.max() > 2 ** i - 1:
            return -i
        budget_data += cellule.size * i

    return budget_data
