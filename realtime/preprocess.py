import numpy as np


def range_fft(data, win):
    if win == 0:
        win = np.blackman(data.shape[0])
    elif win == 1:
        win = np.hanning(data.shape[0])
    else:
        win = np.ones(data.shape[0])
    win = win[:, None, None]
    data = data * win
    r_fft = np.fft.fft(data, axis=0)
    return r_fft


def doppler_fft(data, win):
    if win == 0:
        win = np.blackman(data.shape[1])
    elif win == 1:
        win = np.hanning(data.shape[1])
    else:
        win = np.ones(data.shape[1])
    win = win[None, :, None]
    data = data * win
    data = np.fft.fft(data, axis=1)
    data = np.fft.fftshift(data, axes=1)
    return data


def angele_fft(data):
    data = np.fft.fft(data, axis=-1, n=64)
    data = np.fft.fftshift(data, axes=-1)
    return data


def remove_static_target(range_doppler, thr=3):
    n_doppler_bin = range_doppler.shape[1]
    zero_doppler_ind = n_doppler_bin // 2
    range_doppler[:, n_doppler_bin:, zero_doppler_ind-thr:zero_doppler_ind+thr, :] = 0
    return range_doppler


def get_rai(radar_data_cube, win=0, static_thr=3, doppler_thr_ratio=0.5):
    range_cube = range_fft(radar_data_cube, win)
    range_doppler_cube = doppler_fft(range_cube, win)
    range_doppler_cube = remove_static_target(range_doppler_cube, thr=static_thr)
    doppler = np.mean(range_doppler_cube, axis=-1)
    doppler = np.sum(doppler, axis=0)
    thr = np.max(doppler) * doppler_thr_ratio
    range_doppler_angle_cube = angele_fft(range_doppler_cube)
    mask = doppler > thr
    range_doppler_angle_cube[:, mask, :] = 0
    range_angle_image = np.sum(range_doppler_angle_cube, axis=1)

    return range_angle_image