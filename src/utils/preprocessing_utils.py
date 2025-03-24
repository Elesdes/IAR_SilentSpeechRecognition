import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch, stft
from sklearn.decomposition import FastICA
import pywt
from typing import Tuple


def bandpass_filter(
    data: np.ndarray,
    lowcut: float = 20.0,
    highcut: float = 500.0,
    fs: float = 1000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to the signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def notch_filter(
    data: np.ndarray, freq: float = 50.0, fs: float = 1000.0, quality: float = 30.0
) -> np.ndarray:
    """Apply a notch filter to remove power-line interference."""
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data)


def rectify_signal(data: np.ndarray) -> np.ndarray:
    """Rectify the EMG signal (take absolute value)."""
    return np.abs(data)


def smooth_signal(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Smooth the signal using a moving average filter."""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="same")


def apply_ica(data: np.ndarray, n_components: int = None) -> np.ndarray:
    """Apply Independent Component Analysis to remove artifacts."""
    ica = FastICA(n_components=n_components)
    return ica.fit_transform(data)


def wavelet_denoise(
    data: np.ndarray, wavelet: str = "db4", level: int = 4
) -> np.ndarray:
    """Denoise the signal using wavelet transform."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Dividing by 0.6745 to convert Median Absolute Deviation (MAD) to an estimate of standard deviation assuming a Gaussian distribution
    threshold = np.sqrt(2 * np.log(len(data))) * (
        np.median(np.abs(coeffs[-level])) / 0.6745
    )
    coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


# Feature Extraction Methods
def mean_absolute_value(data: np.ndarray) -> float:
    """Calculate the mean absolute value of the signal."""
    return np.mean(np.abs(data))


def zero_crossings(data: np.ndarray, threshold: float = 0.01) -> int:
    """Count zero crossings exceeding a threshold."""
    zero_crosses = np.where(np.diff(np.signbit(data)))[0]
    return len([i for i in zero_crosses if np.abs(data[i] - data[i + 1]) >= threshold])


def waveform_length(data: np.ndarray) -> float:
    """Calculate the total waveform length."""
    return np.sum(np.abs(np.diff(data)))


def slope_sign_changes(data: np.ndarray, threshold: float = 0.01) -> int:
    """Count slope sign changes in the signal."""
    ssc = np.sum(((data[1:-1] - data[:-2]) * (data[1:-1] - data[2:])) > threshold)
    return int(ssc)


def willison_amplitude(data: np.ndarray, threshold: float = 0.01) -> int:
    """Count Willison amplitude changes exceeding the threshold."""
    return int(np.sum(np.abs(np.diff(data)) > threshold))


def simple_square_integral(data: np.ndarray) -> float:
    """Calculate the simple square integral."""
    return np.sum(data**2)


def median_frequency(data: np.ndarray, fs: float = 1000.0) -> float:
    """Calculate median frequency from power spectral density."""
    freqs, psd = welch(data, fs)
    cumulative_sum = np.cumsum(psd)
    median_freq = freqs[np.where(cumulative_sum >= cumulative_sum[-1] / 2)[0][0]]
    return median_freq


def mean_frequency(data: np.ndarray, fs: float = 1000.0) -> float:
    """Calculate mean frequency from power spectral density."""
    freqs, psd = welch(data, fs)
    return np.sum(freqs * psd) / np.sum(psd)


def power_spectrum_density(
    data: np.ndarray, fs: float = 1000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the power spectrum density of the signal."""
    return welch(data, fs)


def short_time_fourier_transform(
    data: np.ndarray, fs: float = 1000.0, window: str = "hann", nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform of the signal."""
    return stft(data, fs, window=window, nperseg=nperseg)


def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calculate sample entropy of the signal."""
    N = len(data)

    def _phi(m_val):
        x = np.array([data[i : i + m_val] for i in range(N - m_val + 1)])
        C = np.sum([np.sum(np.abs(x - xi).max(axis=1) <= r) - 1 for xi in x])
        return C / (N - m_val + 1)

    return -np.log(_phi(m + 1) / _phi(m))


def approximate_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calculate approximate entropy of the signal."""
    N = len(data)

    def _phi(m_val):
        x = np.array([data[i : i + m_val] for i in range(N - m_val + 1)])
        C = np.sum([np.sum(np.abs(x - xi).max(axis=1) <= r) for xi in x]) / (
            N - m_val + 1
        )
        return np.log(C)

    return _phi(m) - _phi(m + 1)
