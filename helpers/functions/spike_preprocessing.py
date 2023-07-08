import numpy as np
from scipy import signal


def spike_band_filtering(y: np.ndarray, sample_rate: int, low: int, high: int, order: int=4) -> np.ndarray:
    # the nyquist frequency
    nyquist = 0.5 * sample_rate

    # form upper/lower frequency bounds
    lower = low / nyquist
    upper = high / nyquist

    # filter the data using the butterworth filter
    b, a = signal.butter(order, [lower, upper], btype='band')

    # this compensates for phase shifting as it does forward and backwards pass
    y_filtered = signal.filtfilt(b, a, y, padlen=len(y)-1)

    return y_filtered
