import numpy as np


def classify_dynamic_frame_background_diff(rai, mov):
    x


def classify_dynamic_frame(normalized_data):
    # input: max-min normalized DRAI
    maxv = normalized_data.max()
    index = np.unravel_index(normalized_data.argmax(), normalized_data.shape)
    if index[1] > 3 and index[1] < 29:
        m1 = np.mean(normalized_data[0:index[1] - 3, :])
        m2 = np.mean(normalized_data[index[1] + 3:32, :])
        m = (m1 + m2) / 2
    elif index[1] < 3:
        m = np.mean(normalized_data[6:32, :])
    else:
        m = np.mean(normalized_data[0:29, :])
    a = np.log(float(maxv) / float(m) + 1)

    if a < 1.8:
        label = 0
    else:
        label = 1

    return label