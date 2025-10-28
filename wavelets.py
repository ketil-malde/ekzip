import numpy as np
import pywt
import zlib

CAST = np.float16


def compress1(signal, wavelet='db4', level=4, threshold_ratio=0.10):
    '''Apply wavelet compression to a real-valued vector'''
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='periodic')
    cshapes = [c.shape for c in coeffs]
    coeffs_flat = np.concatenate([c for c in coeffs])
    threshold = np.percentile(np.abs(coeffs_flat), 100 * (1 - threshold_ratio))
    comp = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    compressed_data = zlib.compress(np.concatenate(comp).astype(CAST).tobytes())

    return compressed_data, wavelet, level, cshapes


def decompress1(compressed_data, wavelet, level, shapes):
    '''Decompress data to a real-valued vector'''
    compr_flat = np.frombuffer(zlib.decompress(compressed_data), dtype=CAST)
    comp = []
    start = 0
    for shape in shapes:
        comp.append(compr_flat[start:start + shape].reshape(shape))
        start += shape

    recon = pywt.waverec(comp, wavelet, mode='periodic')
    return recon


def compress(signal, wavelet='db4', level=4, threshold_ratio=0.10):
    # Apply wavelet transform to get sets of coefficient vectors
    coeffs_real = pywt.wavedec(signal.real, wavelet, level=level, mode='periodic')
    coeffs_imag = pywt.wavedec(signal.imag, wavelet, level=level, mode='periodic')
    coeffs_shapes = [c.shape for c in coeffs_real]
    coeffs_shapes2 = [c.shape for c in coeffs_imag]
    assert coeffs_shapes == coeffs_shapes2, f'Shapes: {coeffs_shapes}, {coeffs_shapes2}'

    # Flatten coefficients and calculate threshold
    coeffs_flat = np.concatenate([c for c in coeffs_real] + [c for c in coeffs_imag])
    threshold = np.percentile(np.abs(coeffs_flat), 100 * (1 - threshold_ratio))

    # Threshold original coefficient arrays (not coeffs_flat)
    comp_real = [pywt.threshold(c, threshold, mode='soft') for c in coeffs_real]
    comp_imag = [pywt.threshold(c, threshold, mode='soft') for c in coeffs_imag]

    comp_real_flat = np.concatenate(comp_real).astype(CAST)
    comp_imag_flat = np.concatenate(comp_imag).astype(CAST)
    compressed_data = [
        zlib.compress(comp_real_flat.tobytes()),
        zlib.compress(comp_imag_flat.tobytes())
    ]

    return compressed_data, wavelet, level, coeffs_shapes


def decompress(compressed_data, wavelet, level, shapes):
    comp_real_flat = np.frombuffer(zlib.decompress(compressed_data[0]), dtype=CAST)
    comp_imag_flat = np.frombuffer(zlib.decompress(compressed_data[1]), dtype=CAST)

    # Reshape flattened coefficients back into lists of arrays
    comp_real = []
    comp_imag = []
    start = 0
    for shape in shapes:
        size = shape[0]
        comp_real.append(comp_real_flat[start:start + size].reshape(shape))
        comp_imag.append(comp_imag_flat[start:start + size].reshape(shape))
        start += size

    # Reconstruct real and imaginary parts
    recon_real = pywt.waverec(comp_real, wavelet, mode='periodic')
    recon_imag = pywt.waverec(comp_imag, wavelet, mode='periodic')

    # Combine into complex signal
    return recon_real + 1j * recon_imag

