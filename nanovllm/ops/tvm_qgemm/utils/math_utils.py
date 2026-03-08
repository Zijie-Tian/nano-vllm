import numpy as np


def get_bits_alphas(bits: int):
    alphas = [1 / 2, 1, 2, 4, 8, 16, 32, 64]
    return alphas[:bits]


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Normalized Mean Square Error (NMSE).

    Args:
        a: Original signal
        b: Quantized/reconstructed signal

    Returns:
        NMSE value (0 is perfect, 1 is 100% error)
    """
    a, b = a.astype(np.float32), b.astype(np.float32)
    return np.mean(np.square(a - b)) / np.mean(np.square(a))


def compute_sqnr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Signal-to-Quantization-Noise Ratio (SQNR) in dB.

    Args:
        a: Original signal
        b: Quantized/reconstructed signal

    Returns:
        SQNR in decibels (higher is better)
    """
    a, b = a.astype(np.float32), b.astype(np.float32)
    # Avoid log(0) by adding small epsilon
    noise_power = np.sum(np.square(a - b))
    if noise_power == 0:
        return float('inf')  # Perfect reconstruction
    return 10 * np.log10(np.sum(np.square(a)) / noise_power)