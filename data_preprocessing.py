import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def low_pass_filter(x):
    f_order = 4
    f_cutoff = 300
    f_sampling = 1000
    f_nyquist = 0.5*f_sampling
    normlise_cutoff = f_cutoff/f_nyquist

    b, a = signal.butter(f_order, normlise_cutoff, btype='low', analog=False)
    filtered_x = signal.filtfilt(b, a, x)
    return filtered_x

def apply_filter(X):
    X_filtered = np.zeros_like(X)
    for i in range(8):
        X_filtered[i, :] = low_pass_filter(X[i, :])
    return X_filtered

def get_recording_idx(df):
    raw_data = []
    for class_idx in range(1, 8):
        class_data = []
        class_bool = df.loc[:, 'class'].values == int(class_idx)
        class_df = df[class_bool]
        for label_idx in range(1, 37):
            label_data = []
            label_index = class_df[class_df['label'] == label_idx].index.values.tolist()
            if len(label_index) != 0: 
                sample = []       
                for idx, _ in enumerate(label_index):
                    if idx == 0:
                        sample.append(label_index[0])
                    else:
                        if label_index[idx] - label_index[idx-1] <= 100:
                            sample.append(label_index[idx])
                        else:
                            label_data.append(sample)
                            sample = [label_index[idx]]
                class_data.append(label_data)
            else:
                class_data.append(label_data)
        raw_data.append(class_data)
    return raw_data

# add option for people selection
def get_recording(df, filter_flag):
    idxs = get_recording_idx(df)
    raw_recordings = []
    for class_idx in range(7):
        class_recordings = []
        for label_idx in range(36):
            idx = idxs[class_idx][label_idx]
            label_recordings = []
            if len(idx) != 0:
                for sample_idx in idx:
                    X = df.iloc[sample_idx, 1:9].values.T
                    if filter_flag:
                        X = apply_filter(X)
                    label_recordings.append(X)
            class_recordings.append(label_recordings)
        raw_recordings.append(class_recordings)
    return raw_recordings