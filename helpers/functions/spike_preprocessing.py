import numpy as np
from scipy import signal


def spike_band_filtering(y: np.ndarray, sample_rate: int, low: int, high: int, order: int=4) -> np.ndarray:
    """
    Filters the input signal using a Butterworth bandpass filter.

    Parameters
    ----------
    y : np.ndarray
        The input signal.
    sample_rate : int
        The sample rate of the signal in Hz.
    low : int
        The lower frequency bound for the bandpass filter in Hz.
    high : int
        The upper frequency bound for the bandpass filter in Hz.
    order : int, optional
        The order of the Butterworth filter. Default is 4.

    Returns
    -------
    y_filtered : np.ndarray
        The filtered signal.

    Notes
    -----
    This function uses a forward and backward pass filter (filtfilt) to avoid phase shift. 
    The padlen argument is set to len(y)-1 to ensure that the entire signal is filtered.
    """

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