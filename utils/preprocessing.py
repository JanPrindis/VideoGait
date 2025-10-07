import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sympy.physics.quantum.identitysearch import scipy

def cubic_interpolate_nan(y):
    y = np.array([np.nan if v is None else v for v in y], dtype=float)
    mask = ~np.isnan(y)
    if np.sum(mask) < 2:
        return y # Not enough points to interpolate

    first_valid = np.argmax(mask)
    last_valid = len(y) - np.argmax(mask[::-1]) - 1

    x = np.arange(len(y))
    x_seg = x[first_valid:last_valid + 1]
    y_seg = y[first_valid:last_valid + 1]

    mask_seg = ~np.isnan(y_seg)
    cs = scipy.interpolate.CubicSpline(x_seg[mask_seg], y_seg[mask_seg])
    y_interp = cs(x_seg)

    return np.array(y_interp)

def butterworth_filter(data, cutoff=5, fs=60.0, order=5):
    normal_cutoff = cutoff / (fs / 2)
    b, a = butter(order, normal_cutoff, btype='low')
    y = filtfilt(b, a, data)
    return np.array(y)

def find_minima_maxima(data, distance=20, prominence=5):
    minima, _ = find_peaks(-data, distance=distance, prominence=prominence)
    maxima, _ = find_peaks(data, distance=distance, prominence=prominence)
    minima_out = [None] * len(data)
    maxima_out = [None] * len(data)
    for p in minima:
        minima_out[p] = data[p]
    for p in maxima:
        maxima_out[p] = data[p]
    return np.array(minima_out), np.array(maxima_out)
