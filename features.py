import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import yule_walker

def mean_absolute_value(x):
    return np.mean(np.abs(x), axis=0)

def average_amplitude_change(x):
    diff = np.abs(np.diff(x))
    return np.sum(diff)/len(x)

def zero_crossings(x):
    return np.sum(np.diff(np.sign(x)) != 0, axis=0)

def root_mean_square(x):
    return np.sqrt(np.mean(x**2, axis=0))

def slope_sign_changes(x):
    return np.sum(np.diff(np.sign(np.diff(x))) != 0, axis=0)

def autoregressive_coefficients(x):
    rho, _ = yule_walker(x, order=4) 
    return rho

def cepstral_coefficients(x):
    model = sm.tsa.AutoReg(x, lags=4).fit()
    ar_coeffs = model.params[1:]
    cepstral_coeffs = np.zeros(4)
    cepstral_coeffs[0] = -ar_coeffs[0]
    for p in range(2, 5):
        sum_term = sum((1 - l/p) * ar_coeffs[p-l-1] * cepstral_coeffs[l-1] for l in range(1, p))
        cepstral_coeffs[p-1] = -ar_coeffs[p-1] - sum_term
    return cepstral_coeffs

def myopulse_percentage_rate(x, threshold=3e-5):
    myop_output = np.where(np.abs(x) >= threshold, 1, 0)
    return np.sum(myop_output)/len(x)

def mean_frequency(window, fs=1000):
    freqs = np.fft.fftfreq(len(window), 1/fs)
    fft_values = np.fft.fft(window, axis=0)
    psd = np.abs(fft_values)**2
    return np.sum(freqs * psd, axis=0) / np.sum(psd, axis=0)

def median_frequency(x, fs=1000):
    fft_values = np.abs(np.fft.fft(x, axis=0))**2
    cumulative_power = np.cumsum(fft_values, axis=0)
    total_power = cumulative_power[-1]
    return np.argmax(cumulative_power >= 0.5 * total_power, axis=0)

def extract_features(x):
    features = []
    features.append(mean_absolute_value(x))
    features.append(average_amplitude_change(x))
    features.append(zero_crossings(x))
    features.append(root_mean_square(x))
    features.append(slope_sign_changes(x))
    features.append(autoregressive_coefficients(x)[0])
    features.append(autoregressive_coefficients(x)[1])
    features.append(autoregressive_coefficients(x)[2])
    features.append(autoregressive_coefficients(x)[3])
    features.append(cepstral_coefficients(x)[0])
    features.append(cepstral_coefficients(x)[1])
    features.append(cepstral_coefficients(x)[2])
    features.append(cepstral_coefficients(x)[3])
    features.append(myopulse_percentage_rate(x))
    features.append(mean_frequency(x))
    features.append(median_frequency(x))
    feature_names = ['MAV', 'AAC', 'ZC', 'RMS', 'SSC', 'AC1', 'AC2', 'AC3', 'AC4', 'CC1', 'CC2', 'CC3', 'CC4', 'MYPR', 'MF', 'MDF']
    return np.array(features).astype(np.float64), feature_names