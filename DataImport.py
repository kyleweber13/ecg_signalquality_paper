import nimbalwear
import pyedflib
import pandas as pd
import os


def read_header(edf_file: str):
    """ Function that calls pyedflib to read sample rate, file duration, and
        start timestamp from header of EDF file without importing any data

        Parameters
        ----------
        edf_file
            pathway to EDF file

        Returns
        -------
        dictionary
            keys: ["sample_rate"], ["duration"], and ["startdate"]
    """

    print(f"\nChecking {edf_file} for header info...")

    f = pyedflib.EdfReader(edf_file)
    fs = f.getSampleFrequency(0)
    dur = f.getFileDuration()
    start = f.getStartdatetime()

    f.close()

    return {"sample_rate": fs, 'duration': dur, 'start_time': start}


def import_snr_raw(full_id: str,
                   snr_edf_folder: str = "W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/",
                   edf_folder: str = 'W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/',
                   check_header: bool = True,
                   df_subj: pd.DataFrame = None):
    """ Reads in SNR timeseries data from EDF and generates timestamps

        Parameters
        ----------
        full_id
            participant ID not including collection ID
        snr_edf_folder
            pathway to folder containing SNR EDF files
        edf_folder
            pathway to folder containing ECG EDF files
        check_header
            boolean to call ProcessData.read_header() to get start time, sample rate, collection duration
        df_subj
            dataframe from ProcessData.get_collection_details()

        Returns
        -------
        dataframe of collection details
        SNR timeseries data
        SNR timestamps
    """

    snr_data = nimbalwear.Device()
    snr_data.import_edf(f"{snr_edf_folder}/{full_id}_01_snr.EDF")

    if check_header:
        d = read_header(edf_file=f"{edf_folder}{full_id}_01_BF36_Chest.edf")
        d['days'] = [None]
    if not check_header:
        d = {'sample_rate': [None], 'days': [None], 'start_time': [None]}

    if df_subj is not None and not check_header:
        d = df_subj.loc[df_subj['full_id'] == full_id].iloc[0]

    ts = pd.date_range(start=d['start_time'], freq="{}ms".format(1000 / d['sample_rate']),
                       periods=len(snr_data.signals[0]))

    df_out = pd.DataFrame({"full_id": [full_id], 'sample_rate': [d['sample_rate']],
                           'days': [d['days']], 'start_time': [d['start_time']]})

    return df_out, snr_data.signals[snr_data.get_signal_index('snr')], ts


def import_bittium(filepath: str):
    """ Calls nimbalwear.Device() to import given ECG EDF file

        Parameters
        ----------
        filepath
            pathway to ECG EDF file

        Returns
        -------
        nimbalwear.Device() instance
    """

    ecg = nimbalwear.Device()
    ecg.import_edf(filepath)

    start_time = ecg.header['startdate'] if 'startdate' in ecg.header.keys() else ecg.header['start_datetime']
    ecg.fs = ecg.signal_headers[ecg.get_signal_index('ECG')]['sample_rate']

    ecg.ts = pd.date_range(start=start_time, periods=len(ecg.signals[ecg.get_signal_index('ECG')]),
                           freq=f"{1000 / ecg.fs}ms")

    return ecg


def import_nonwear_log(file: str,
                       full_id: str or None = None):
    """ Imports nonwear bout csv file, formats participant ID, and crops df to include only specified participant

        Parameters
        ----------
        file
            pathway to nonwear file
        full_id
            participant ID. If None, full df is returned. If given, only rows with that ID are returned

        Returns
        -------
        dataframe
    """

    df = pd.read_excel(file)

    df['full_id'] = [f"{row.study_code}_{row.subject_id}" for row in df.itertuples()]

    if full_id is not None:
        df = df.loc[df['full_id'] == full_id].reset_index(drop=True)

    return df


def import_posture(filepath: str):
    """ Imports posture csv file and formats any time-related columns.

        Parameter
        ---------
        filepath
            pathway to posture file

        Returns
        -------
        dataframe
    """

    df_posture = pd.read_csv(filepath)

    for col in df_posture.columns:
        if 'time' in col or 'Time' in col:
            df_posture[col] = pd.to_datetime(df_posture[col])

    df_posture.columns = ['start_time', 'posture', 'ant_ang', 'up_ang', 'left_ang']

    return df_posture


def combine_files(folder: str):
    """ Combines all dataframe files in folder into single dataframe

        Parameter
        ---------
        folder
            pathway to folder

        Returns
        -------
        combined dataframe
    """

    files = os.listdir(folder)
    df = pd.read_csv(folder + files[0])

    for file in files[1:]:
        df = pd.concat([df, pd.read_csv(folder + file)])

    return df

