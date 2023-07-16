import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# this algorithm is a modification of the peakutils.peak.indexes function

def detect_spikes(y: np.ndarray, threshold: float, minimum_gap: int=1, use_absolute_threshold: bool=False, flipped: bool=False) -> np.ndarray:
    """
    Detects spikes (or troughs) in a given signal.

    Parameters
    ----------
    y: np.ndarray
        The input signal.
    threshold: float
        The threshold value for spike detection. If `use_absolute_threshold` is False, this is a relative value.
    minimum_gap: int, optional
        The minimum number of samples between spikes. Default is 1.
    use_absolute_threshold : bool, optional
        If True, `threshold` is an absolute value. If False, `threshold` is a relative value. Default is False.
    flipped: bool, optional
        If True, the function will detect troughs (downward spikes) instead of peaks (upward spikes). Default is False.

    Returns
    -------
    np.ndarray
        An array of indices in `y` where spikes were detected.

    Raises
    ------
    ValueError
        If `y` is an unsigned array.

    Notes
    -----
    This function uses a first order difference method to detect spikes. It first computes the first differential of `y`, then finds the indices where the differential changes sign (indicating a peak or trough). It then filters these indices based on the `threshold` value and the `minimum_gap` between spikes.

    If `flipped` is True, the function detects troughs instead of peaks. This is done by reversing the sign of the differential and the `threshold` value.

    The function returns an array of indices in `y` where spikes (or troughs) were detected.
    """

    # Check if y is unsigned array
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")
    
    # Convert relative threshold to absolute if necessary
    if not use_absolute_threshold:
        threshold = threshold * (np.max(y) - np.min(y)) + np.min(y)

    # Compute the first differential
    dy = np.diff(y)

    # Propagate left and right values successively to fill all plateau pixels (no gradient)
    zeros, = np.where(dy == 0)

    # Check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([], dtype=np.int64)
    
    # Find the peaks or troughs
    if flipped:
        extrema = np.where((np.hstack([dy, 0.0]) > 0.0) & (np.hstack([0.0, dy]) < 0.0) & (np.less(y, threshold)))[0]
    else:
        extrema = np.where((np.hstack([dy, 0.0]) < 0.0) & (np.hstack([0.0, dy]) > 0.0) & (np.greater(y, threshold)))[0]

    # Handle multiple peaks or troughs, respecting the minimum distance
    if extrema.size > 1 and minimum_gap > 1:
        sorted_extrema = extrema[np.argsort(y[extrema])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[extrema] = False

        for extremum in sorted_extrema:
            if not rem[extremum]:
                sl = slice(max(0, extremum - minimum_gap), extremum + minimum_gap + 1)
                rem[sl] = True
                rem[extremum] = False

        extrema = np.arange(y.size)[~rem]

    return extrema


def merge_spike_indices(spike_indices: np.ndarray[np.ndarray], tolerance: int=5) -> np.ndarray:
    """
    Merge spike indices from multiple channels into a single array. 
    If the points are close together only the midpoint is added.

    Parameters:
    -----------
    spike_indices: list of numpy arrays
        Each numpy array contains spike indices for a single channel.
    tolerance: int, optional
        The maximum distance between spike indices that will be considered as the same spike.
        Indices within this distance will be replaced by their midpoint.

    Returns:
    --------
    numpy array
        A single array of merged spike indices.
    """

    # flatten all indices into a single list
    all_indices = np.concatenate(spike_indices)
    
    # sort the indices
    all_indices.sort()
    
    # initialise the output list with the first index
    merged_indices = [all_indices[0]]
    
    # go through the sorted list and merge indices that are close together
    for index in all_indices[1:]:
        if index - merged_indices[-1] <= tolerance:
            # if the current index is close to the last one in the output list, replace the last one with their average (rounded to nearest integer)
            merged_indices[-1] = round((merged_indices[-1] + index) / 2)
        else:
            # if the current index is not close to the last one, add it to the output list
            merged_indices.append(index)
    
    return np.array(merged_indices, dtype=int)


def get_waveforms(y: np.ndarray, spike_indices: np.ndarray, duration: int, sample_rate: int, window_shift_ratio: float=0.5) -> Tuple[np.ndarray[np.ndarray], List[dict]]:
    """
    Extracts waveforms from a signal at given indices.

    Parameters
    ----------
    y : np.ndarray
        The input signal.
    spike_indices : np.ndarray
        The indices in `y` where spikes were detected.
    duration : int
        The duration of the waveform in milliseconds.
    sample_rate : int
        The sample rate of the signal in Hz.
    window_shift_ratio : float, optional
        The ratio of the window size to shift the window to the left of the spike. Default is 0.5.

    Returns
    -------
    waveforms : np.ndarray
        A nested numpy array of extracted waveforms.
    waveform_info : list of dict
        A list of dictionaries containing information about each extracted waveform.

    Notes
    -----
    The dictionaries of waveform_info contain the starting and finishing index of the waveform, its greatest positive 
    and negative amplitudes and the values of the waveform (corresponding to the data in `waveforms`).
    
    """

    # Calculate the number of samples to extract around each spike
    window_size = int(sample_rate * duration / 1000)

    # Calculate the number of samples to shift the window
    shift = int(window_size * window_shift_ratio)

    waveforms = []
    waveform_info = []

    # Iterate over the spike indices
    for i in spike_indices:

        # Calculate the start and end of the window
        start = int(i - shift)
        end = int(start + window_size)

        # Extract the waveform
        waveform = y[start:end]

        # Append the waveform to the list
        waveforms.append(waveform)

        # Store information about the waveform
        spike_info = {
            'spike_start': start,
            'spike_end': end,
            'lowest_value': np.min(waveform),
            'highest_value': np.max(waveform),
            'values': waveform
        }

        waveform_info.append(spike_info)

    # Convert the lists to numpy arrays
    waveforms = np.array(waveforms)

    return waveforms, waveform_info
