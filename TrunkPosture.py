import nimbalwear
import pandas as pd
import numpy as np
from scipy import signal
from datetime import timedelta as td


def prep_accelerometer_df(x, y, z, sample_rate, start_time):
    """
    Creates the proper dataframe format for an individual accelerometer for the remainder of the posture analysis
    """
    # Get data
    data_dict = {
        "Anterior": [-i for i in z],
        "Up": x,
        "Left": y,
        "start_time": start_time,
        "sample_rate": sample_rate}

    # Epoch data
    ts = pd.date_range(data_dict['start_time'], periods=len(data_dict['Anterior']),
                       freq=(str(1 / data_dict['sample_rate']) + 'S'))
    df = pd.DataFrame({'start_time': ts, 'ant_acc': data_dict['Anterior'],
                       'up_acc': data_dict['Up'], 'left_acc': data_dict['Left']})
    return df.resample(
        f'{1}S', on='start_time').mean().reset_index(col_fill='start_time')  # Resample to 1s intervals using mean


def _filter_acc_data(x, y, z):
    """Filter accelerometer data to obtain gravity and body movement components
    Args:
        x: time array of the single-axis acceleration values
        y: time array of the single-axis acceleration values
        z: time array of the single-axis acceleration values
    Returns:
        - list of gravity components of each axis of accelerometer data
        - list of body movement components of each axis of accelerometer data
    """
    # median filter to remove high-frequency noise spikes
    denoised_x = signal.medfilt(x)
    denoised_y = signal.medfilt(y)
    denoised_z = signal.medfilt(z)

    # low pass elliptic filter to obtain gravity component
    sos = signal.ellip(3, 0.01, 100, 0.25, 'low', output='sos')
    grav_x = signal.sosfilt(sos, denoised_x)
    grav_y = signal.sosfilt(sos, denoised_y)
    grav_z = signal.sosfilt(sos, denoised_z)

    # subtract gravity component from signal to obtain body movement component
    bm_x = denoised_x - grav_x
    bm_y = denoised_y - grav_y
    bm_z = denoised_z - grav_z

    return [grav_x, grav_y, grav_z], [bm_x, bm_y, bm_z]


def _get_angles(grav_x, grav_y, grav_z):
    """Get angles between each axis using the gravity components of the accelerometer data created in _filter_acc_data
    Args:
        grav_x: time array of the gravity component of a single-axis of an accelerometer
        grav_y: time array of the gravity component of a single-axis of an accelerometer
        grav_z: time array of the gravity component of a single-axis of an accelerometer
    Returns:
        angle_x: time array of angles for a single-axis
        angle_y: time array of angles for a single-axis
        angle_z: time array of angles for a single-axis
    """
    magnitude = np.sqrt(np.square(np.array([grav_x, grav_y, grav_z])).sum(axis=0))
    angle_x = np.arccos(grav_x / magnitude) * 180 / np.pi
    angle_y = np.arccos(grav_y / magnitude) * 180 / np.pi
    angle_z = np.arccos(grav_z / magnitude) * 180 / np.pi

    return angle_x, angle_y, angle_z


def create_posture_df_template(data_df):
    """Classify static movement time points as transition based on a threshold. This posture dataframe is then used for
     input into the wrist/ankle/chest/thigh posture classifiers.
    Args:
        data_df: dataframe with timestamps and 3 axes of accelerometer data;
                 must have columns: timestamp, ant_acc, up_acc, left_acc;
                 can be created using prep_accelerometer_df
        gait_mask: time array of 1s and 0s indicating gait activity
        tran_type: optional; string indicating type of transition to use.
                   Options are 'jerk' and 'ang_vel'
        tran_thresh: optional; float or integer indicating the threshold to
                     define a period as a transition
    Returns:
        posture_df: dataframe of timestamps, preliminary postures, angle data,
                 gait mask, and transitions
    """
    # obtain the various components of the posture dataframe
    grav_data, bm_data = _filter_acc_data(
        data_df['ant_acc'], data_df['up_acc'], data_df['left_acc'])
    ant_ang, up_ang, left_ang = _get_angles(*grav_data)

    # create the posture dataframe
    posture_df = pd.DataFrame(
        {'start_time': data_df['start_time'],
         'posture': 'other',
         'ant_ang': ant_ang, 'up_ang': up_ang, 'left_ang': left_ang})

    return posture_df


def classify_chest_posture(posture_df):
    """Determine posture based on chest angles from each axis. TODO: Add classification details to external documentation
    Posture can be: sitstand, reclined, prone, supine, leftside, rightside, or
    other. Angle interpretations for posture classification:
        0 degrees: positive axis is pointing upwards (against gravity)
        90 degrees: axis is perpendicular to gravity
        180 degrees: positive axis is pointing downwards (with gravity)
    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template
    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    # sit/stand: static with up axis upwards
    posture_df.loc[(posture_df['up_ang'] < 45), 'posture'] = "sitstand"

    # prone: static with anterior axis downwards, left axis horizontal
    posture_df.loc[(135 <= posture_df['ant_ang']) &
                   (45 <= posture_df['left_ang']) & (posture_df['left_ang'] <= 135),
                   'posture'] = "prone"

    # supine: static with anterior axis upwards, left & up axes horizontal
    posture_df.loc[(posture_df['ant_ang'] <= 45) &
                   (70 <= posture_df['up_ang']) & (45 <= posture_df['left_ang']) &
                   (posture_df['left_ang'] <= 135), 'posture'] = "supine"

    # reclined: static with anterior axis upwards, left axis horizontal,
    # up axis above horizontal
    posture_df.loc[(posture_df['ant_ang'] <= 70) &
                   (45 <= posture_df['up_ang']) & (posture_df['up_ang'] < 70) &
                   (45 <= posture_df['left_ang']) & (posture_df['left_ang'] <= 135),
                   'posture'] = "reclined"

    # left side: static with left axis downwards, up axis horizontal
    posture_df.loc[(45 <= posture_df['up_ang']) & (posture_df['up_ang'] <= 135) &
                   (135 <= posture_df['left_ang']), 'posture'] = "leftside"

    # right side: static with left axis upwards, up axis horizontal
    posture_df.loc[(45 <= posture_df['up_ang']) & (posture_df['up_ang'] <= 135) &
                   (posture_df['left_ang'] < 45), 'posture'] = "rightside"

    return posture_df


def calculate_chest_posture(chest_x=None, chest_y=None, chest_z=None, chest_freq=None, chest_start=None):

    prepped_df = prep_accelerometer_df(chest_x, chest_y, chest_z, sample_rate=chest_freq, start_time=chest_start)

    df_temp = create_posture_df_template(prepped_df)

    df = classify_chest_posture(df_temp)

    df['posture'].replace({"sitstand": 'upright'}, inplace=True)

    return df


def bout_posture(df):

    postures = [df.iloc[0]['posture']]
    start_times = [df.iloc[0]['start_time']]
    end_times = [df.iloc[1]['start_time']] if df.iloc[0]['posture'] != df.iloc[1]['posture'] else []

    for row in df.iloc[1:, :].itertuples():

        if row.posture == postures[-1]:
            pass

        if row.posture != postures[-1]:
            postures.append(row.posture)
            start_times.append(row.start_time)
            end_times.append(row.start_time)

    if len(start_times) > len(end_times):
        end_times.append(df.iloc[-1]['start_time'] + td(seconds=1))

    df_out = pd.DataFrame({'start_time': start_times, 'end_time': end_times, 'posture':  postures})
    df_out.insert(loc=2,
                  column='duration',
                  value=[(row.end_time-row.start_time).total_seconds() for row in df_out.itertuples()])

    return df_out


def create_posture_files(full_id: str,
                         edf_folder: str = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
                         save_dir: str = "O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/chest_posture/"):

    ecg = nimbalwear.Device()
    ecg.import_edf(f"{edf_folder}{full_id}_BF36_Chest.edf")

    # 1-second posture classification
    df_post = calculate_chest_posture(chest_x=ecg.signals[ecg.get_signal_index('Accelerometer x')],
                                      chest_y=ecg.signals[ecg.get_signal_index('Accelerometer y')],
                                      chest_z=ecg.signals[ecg.get_signal_index('Accelerometer z')],
                                      chest_freq=ecg.signal_headers[ecg.get_signal_index('Accelerometer x')]['sample_rate'],
                                      chest_start=ecg.header['start_datetime'])

    # start/end times for each posture bout
    df_bouts = bout_posture(df_post)

    df_post.to_csv(f"{save_dir}{full_id}_posture.csv", index=False)
    df_bouts.to_csv(f"{save_dir}{full_id}_posture_bouts.csv", index=False)
