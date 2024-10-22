import librosa
import librosa.display
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import yule_walker
import matplotlib.pyplot as plt

def moments(x):
    n = len(x)
    std_dev = np.std(x, ddof=0)
    # first 4 moments
    first = x.mean()    # mean
    second = x.var()    # variance
    third = (n/((n-1)*(n - 2)))*np.sum(((x-first)/std_dev)**3)  # skew
    fourth = ((n*(n+1))/((n-1)*(n-2)*(n-3)))*np.sum(((x-first)/std_dev)**4)-(3*(n-1)**2/((n-2)*(n-3)))   # kurtosis
    return first, second, third, fourth

def domain_specific(x):
    def cepstral_coefficients(x):
        model = sm.tsa.AutoReg(x, lags=4).fit()
        ar_coeffs = model.params[1:]
        cepstral_coeffs = np.zeros(4)
        cepstral_coeffs[0] = -ar_coeffs[0]
        for p in range(2, 5):
            sum_term = sum((1 - l/p) * ar_coeffs[p-l-1] * cepstral_coeffs[l-1] for l in range(1, p))
            cepstral_coeffs[p-1] = -ar_coeffs[p-1] - sum_term
        return cepstral_coeffs

    mav = np.mean(np.abs(x), axis=0)   # mean absolute value
    aac = np.sum(np.abs(np.diff(x)))/len(x) # average amplitude change
    zc = np.sum(np.diff(np.sign(x)) != 0, axis=0)   # zero-crossing
    rms = np.sqrt(np.mean(x**2, axis=0))    # root mean square
    ssc = np.sum(np.diff(np.sign(np.diff(x))) != 0, axis=0)   # slope sign changes
    ac, _ = yule_walker(x, order=4)   # autoregressive coefficients
    cc = cepstral_coefficients(x)   # cepstral coefficients
    mypr = np.sum(np.where(np.abs(x) >= 3e-5, 1, 0))/len(x)   # myopulse percentage rate
    return mav, aac, zc , rms, ssc, ac[0], ac[1], ac[2], ac[3], cc[0], cc[1], cc[2], cc[3], mypr

def frequency_domain(x):
    def mean_frequency(x, fs=1000):
        freqs = np.fft.fftfreq(len(x), 1/fs)
        fft_values = np.fft.fft(x, axis=0)
        psd = np.abs(fft_values)**2
        return np.sum(freqs * psd, axis=0) / np.sum(psd, axis=0)
    def median_frequency(x, fs=1000):
        fft_values = np.abs(np.fft.fft(x, axis=0))**2
        cumulative_power = np.cumsum(fft_values, axis=0)
        total_power = cumulative_power[-1]
        return np.argmax(cumulative_power >= 0.5 * total_power, axis=0)
    
    mf = mean_frequency(x)
    medf = median_frequency(x)
    return mf, medf

#feature_names = ['MAV', 'AAC', 'ZC', 'RMS', 'SSC', 'AC1', 'AC2', 'AC3', 'AC4', 'CC1', 'CC2', 'CC3', 'CC4', 'MYPR', 'MF', 'MDF']
def extract_signal_features(x):
    m1, m2, m3, m4 = moments(x)
    mav, aac, zc , rms, ssc, ac1, ac2, ac3, ac4, cc1, cc2, cc3, cc4, mypr = domain_specific(x)
    mf, medf = frequency_domain(x)
    return np.array([m1, m2, m3, m4, mav, aac, zc , rms, ssc, ac1, ac2, ac3, ac4, cc1, cc2, cc3, cc4, mypr, mf, medf]).astype(np.float64)

def extract_channel_features(Yi, f_range):
    y_channel = None
    for channel_idx in range(8):
        x_c = Yi[channel_idx]
        info = extract_signal_features(x_c)
        if channel_idx == 0:
            y_channel = info[f_range]
        else:
            y_channel = np.hstack((y_channel, info[f_range]))
    return y_channel

# add option for people selection
def extract_features_from_recordings(X, f_range):
    X_new = []
    person_id = []
    gesture_id = []
    for class_idx in range(7):
        for row_idx in range(36):
            Y = X[class_idx][row_idx]
            if len(Y) != 0:
                for Yi in Y:
                    features = extract_channel_features(Yi, f_range)
                    X_new.append(features)
                    person_id.append(row_idx+1)
                    gesture_id.append(class_idx+1)
    return np.array(X_new), np.array(person_id), np.array(gesture_id)


def pad_or_truncate(X_signal, target_length=1500):
    X_new = []
    for channel_idx in range(8):
        signal = X_signal[channel_idx, :]
        current_length = len(signal)
        if current_length < target_length:
            new_signal = np.pad(signal, (0, target_length-current_length), mode='constant')
        elif current_length > target_length:
            new_signal = np.array(signal[:target_length])
        else:
            new_signal = signal
        X_new.append(new_signal)
    X_new = np.array(X_new)
    return X_new

def spectrogram_visualise(S_db):
    plt.figure(figsize=(10,6))
    librosa.display.specshow(S_db, sr=1000, hop_length=32, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title('Spectrogram of the Signal (librosa)')
    plt.ylim(0, 200)  # Limit to 200 Hz for clarity
    plt.show()

def generate_spectrogram_channels(X, n_fft, hop_length):
    spectrograms = []
    for channel_idx in range(8):
        signal = X[channel_idx, :]
        D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        S_db = S_db[:128, :]
        S_db = S_db[:, :64]
        spectrograms.append(S_db)
    spectrograms = np.stack(spectrograms, axis=-1)
    return spectrograms

def process_all_recordings(X_rec, n_fft, hop_length):
    X_dataset = []
    person_id = []
    gesture_id = []
    for class_idx in range(7):
        for row_idx in range(36):
            Y = X_rec[class_idx][row_idx]
            if len(Y) != 0:
                for Yi in Y:
                    X_new = pad_or_truncate(Yi)
                    X_spec = generate_spectrogram_channels(X_new, n_fft, hop_length)
                    X_dataset.append(X_spec)
                    person_id.append(row_idx+1)
                    gesture_id.append(class_idx+1)
    X_dataset = np.stack(X_dataset)
    return X_dataset, np.array(person_id), np.array(gesture_id)