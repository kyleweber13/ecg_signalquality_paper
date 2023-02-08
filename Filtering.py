from scipy.signal import butter, lfilter, filtfilt, iirnotch


def filter_signal(data, filter_type, low_f=None, high_f=None, notch_f=None, notch_quality_factor=30.0,
                  sample_f=None, filter_order=2):
    """Function that creates bandpass filter to ECG data.
    Required arguments:
    -data: 3-column array with each column containing one accelerometer axis
    -type: "lowpass", "highpass" or "bandpass"
    -low_f, high_f: filter cut-offs, Hz
    -sample_f: sampling frequency, Hz
    -filter_order: order of filter; integer
    """

    nyquist_freq = 0.5 * sample_f

    if filter_type == "lowpass":
        low = low_f / nyquist_freq
        b, a = butter(N=filter_order, Wn=low, btype="lowpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == "highpass":
        high = high_f / nyquist_freq

        b, a = butter(N=filter_order, Wn=high, btype="highpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == "bandpass":
        low = low_f / nyquist_freq
        high = high_f / nyquist_freq

        b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == 'notch':
        b, a = iirnotch(w0=notch_f, Q=notch_quality_factor, fs=sample_f)
        filtered_data = filtfilt(b, a, x=data)

    return filtered_data
