import pyedflib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from Analysis import *
from Preprocessing import combine_df_avm
from TrunkPosture import bout_posture
from DataImport import import_snr_raw, import_posture, import_bittium
from datetime import timedelta


def get_collection_details(df_subjs: pd.DataFrame,
                           edf_folder: str = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
                           nw_folder: str = "",
                           snr_folder: str = "W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/"):
    """ Function that checks EDF files from given participants to retrieve sample rate, start/end times, and looks for
        whether non-wear detection has been run on the Bittium data.

        Parameters
        ----------
        df_subjs
        edf_folder
            pathway to folder that contains EDF files
        nw_folder
            pathway to folder that contains Bittium non-wear files
        snr_folder
            pathway to folder that contains SNR timeseries EDF files

        Returns
        -------
        dataframe
            columns: ['full_id', 'cohort', 'sample_rate', 'start_time', 'end_time', 'days', 'proc_nw']
    """

    durs = []
    starts = []
    ends = []
    days = []
    rates = []
    nw_proc = []
    snr_proc = []

    for i in tqdm(range(df_subjs.shape[0])):
        row = df_subjs.iloc[i, :]

        edf_file = f'{edf_folder}{row.full_id}_01_BF36_Chest.edf'
        if os.path.exists(edf_file):
            f = pyedflib.EdfReader(edf_file)
            fs = f.getSampleFrequency(0)
            dur = f.getFileDuration()
            start = f.getStartdatetime()

            durs.append(dur)
            days.append(dur / 86400)
            starts.append(start)
            ends.append(start + timedelta(seconds=dur))
            rates.append(fs)

            nw_file = f"{nw_folder}{row.full_id}_01_BF36_Chest_NONWEAR.csv"
            nw_exists = os.path.exists(nw_file)
            nw_proc.append(nw_exists)

        if not os.path.exists(edf_file):
            durs.append(None)
            days.append(None)
            starts.append(None)
            ends.append(None)
            rates.append(None)
            nw_proc.append(None)

        snr_proc.append(os.path.exists(f"{snr_folder}{row.full_id}_01_snr.edf"))

    df_subjs['sample_rate'] = rates
    df_subjs['days'] = days
    df_subjs['start_time'] = starts
    df_subjs['end_time'] = ends
    df_subjs['proc_nw'] = nw_proc

    df = df_subjs[['full_id', 'cohort', 'sample_rate', 'start_time', 'end_time', 'days', 'proc_nw']]

    return df


def create_collection_timestamps(df_subjs: pd.DataFrame,
                                 df_timestamps: pd.DataFrame):
    """ Function that formats start/stop times based on file start/stop times and
        flagged electrode replacement timestamps.

        Parameters
        ----------
        df_subjs
            dataframe from get_collection_details()
        df_timestamps
            dataframe from create_collection_timestamps()

        Returns
        -------
        dataframe
    """

    use_starts = []
    use_ends = []

    for row in df_timestamps.itertuples():
        start = None
        end = None

        d = df_subjs.loc[df_subjs['full_id'] == row.full_id]

        if d.shape[0] == 1:
            d = d.iloc[0]
            start = d['start_time']
            end = d['end_time']

            if not pd.isnull(row.start):
                start = row.start
            if not pd.isnull(row.end):
                end = row.end

        use_starts.append(start)
        use_ends.append(end)

    df_timestamps['use_start'] = use_starts
    df_timestamps['use_end'] = use_ends
    df_timestamps['duration'] = [(j-i).total_seconds()/86400 for i, j in zip(df_timestamps['use_start'],
                                                                             df_timestamps['use_end'])]

    return df_timestamps


def average_snr(snr_signal: np.array or list,
                sample_rate: int,
                timestamps: np.array or list,
                n_secs: int = 1):
    """ Averages snr_signal over n_secs-second intervals

        Parameters
        ---------
        snr_signal
            timeseries SNR signal
        sample_rate
            sample rate of snr_signal in Hz
        timestamps
            timestamps of snr_signal
        n_secs
            number of seconds over which averages are calculated

        Returns
        -------
        dataframe
            ['days']: number of days since start timestamp
    """

    n_samples = int(sample_rate*n_secs)
    avg = [np.mean(snr_signal[i:i+n_samples]) for i in np.arange(0, len(snr_signal), n_samples)]

    epoch_timestamps = timestamps[::n_samples]
    days = np.arange(0, len(avg)) / 86400 * n_secs
    df_out = pd.DataFrame({"start_time": epoch_timestamps, 'days': days, 'snr': avg})

    return df_out


def assign_snr_category(df: pd.DataFrame,
                        thresholds: list or tuple = (5, 18)):
    """ Assigns quality category to SNR value give thresholds

        Parameters
        ----------
        df
            dataframe containing 'snr' column
        thresholds
            thresholds corresponding to Q1/2 and Q2/3 classification

        Returns
        -------
        df with "snr_quality" column added
    """

    thresholds = sorted(thresholds)

    df['snr_quality'] = ['Q3'] * df.shape[0]

    df.loc[(df['snr'] >= thresholds[0]) & (df['snr'] < thresholds[1]), 'snr_quality'] = 'Q2'
    df.loc[df['snr'] >= thresholds[1], 'snr_quality'] = 'Q1'

    return df


def create_mask(start_time: str or pd.Timestamp,
                n_rows: int,
                df: pd.DataFrame,
                epoch_len: int):
    """ Given bouted events and timestamps, creates a binary mask of given epoch length marking those events.

        Parameters
        ----------
        start_time
            start time for mask[0]
        n_rows
            length of output mask (~ duration of data / epoch_len)
        df
            event dataframe. Requires "start_time" and "end_time" columns
            e.g. pass in df containing nonwear bouts --> flags mask as 1 if during nonwear period
        epoch_len
            epoch length in seconds corresponding to the time contained within each element of the mask

        Returns
        -------
        mask where 1s indicate time periods that occurred during the events in df
    """

    mask = np.zeros(n_rows)

    start_time = pd.to_datetime(start_time)

    for row in df.itertuples():
        # for files with bouts (start/end times)
        try:
            start_i = int((row.start_time - start_time).total_seconds() / epoch_len)
            end_i = int((row.end_time - start_time).total_seconds() / epoch_len)

            mask[start_i:end_i] = 1

        # for 1-sec event files (just start time; no end)
        except AttributeError:
            start_i = int((row.start_time - start_time).total_seconds() / epoch_len)
            mask[start_i] = 1

    return mask


def reepoch_data(df_epoch1s: pd.DataFrame,
                 new_epoch_len: int,
                 cutpoints: str or None = None,
                 snr_thresh: list or tuple = (5, 18),
                 avm_thresh: int or float = 15,
                 quiet: bool = True):
    """ Given 1-second epoch data, re-calculates data into new epoch length.

        Parameters
        ----------
        df_epoch1s
            df containing 1-second epoched data
        new_epoch_len
            new epoch length in seconds over which values in df_epoch1s are recalculated
        cutpoints
            'Powell' or 'Fraysse'. If specified, recalculates activity intensity using wrist AVM data
        snr_thresh
            SNR thresholds for Q1/2 and Q2/3 boundaries.
        avm_thresh
            wrist AVM threshold used to flag epoch as active/inactive (not related to intensity)
        quiet
            boolean to print progress to console

        Returns
        -------
        df with new epoch length
    """

    if not quiet:
        print(f"\nEpoching to {new_epoch_len}-second epochs...")

    if cutpoints is not None:
        if cutpoints in ['Powell', 'powell']:
            # powell et al., 2016 GENEActiv dominant wrist cutpoints
            cp = {'light': 51 / (30 * 15) * 1000, 'mod': 68 / (30 * 15) * 1000, 'vig': 142 / (30 * 15) * 1000}
        if cutpoints in ['Fraysse', 'fraysse']:
            # Fraysse et al. (2021) GENEActiv dominant wrist cutpoints
            cp = {'light': 62.5, 'mod': 92.5, 'vig': 10000}

    df_out = pd.DataFrame({"start_time": df_epoch1s['start_time'].iloc[::new_epoch_len],
                           'days': df_epoch1s['days'].iloc[::new_epoch_len]})

    df_out.insert(loc=0, column='full_id', value=[df_epoch1s.iloc[0]['full_id']] * df_out.shape[0])
    df_out.insert(loc=3, column='day_int', value=[int(np.floor(i)) for i in df_out['days']])

    # re-averages SNR data
    data = list(df_epoch1s['snr'])
    df_out['snr'] = [np.mean(data[i:i + new_epoch_len]) for i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]

    # assigns quality classification to SNR data given snr_thresh
    df_out = assign_snr_category(df=df_out, thresholds=snr_thresh)

    # flags epochs as useable for HR or not (Q1 or Q2 vs. Q3)
    df_out['snr_hr'] = [row.snr_quality != 'Q3' for row in df_out.itertuples()]

    # re-epoching sleep, sleep-period-time-window, gait, non-wear masks -------------------

    data = list(df_epoch1s['sleep_mask'])
    df_out['sleep_percent'] = [sum(data[i:i + new_epoch_len])*100/new_epoch_len for
                               i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]
    df_out['sleep_any'] = df_out['sleep_percent'] != 0

    data = list(df_epoch1s['sptw_mask'])
    df_out['sptw_percent'] = [sum(data[i:i + new_epoch_len])*100/new_epoch_len for
                              i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]

    data = list(df_epoch1s['gait_mask'])
    df_out['gait_percent'] = [sum(data[i:i + new_epoch_len])*100/new_epoch_len for
                              i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]
    df_out['gait_any'] = df_out['gait_percent'] != 0

    data = list(df_epoch1s['wrist_nw_mask'])
    df_out['wrist_nw_percent'] = [sum(data[i:i + new_epoch_len])*100/new_epoch_len for
                                  i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]

    data = list(df_epoch1s['ankle_nw_mask'])
    df_out['ankle_nw_percent'] = [sum(data[i:i + new_epoch_len])*100/new_epoch_len for
                                  i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]

    data = list(df_epoch1s['chest_nw_mask'])
    try:
        df_out['chest_nw_percent'] = [sum(data[i:i + new_epoch_len])*100/new_epoch_len for
                                      i in np.arange(0, len(data), new_epoch_len)]
    except TypeError:
        df_out['chest_nw_percent'] = [None] * len(df_out)

    # boolean for if all devices were worn
    df_out['all_wear'] = [row.chest_nw_percent + row.ankle_nw_percent + row.wrist_nw_percent == 0 for
                          row in df_out.itertuples()]

    # re-epoching AVM data from all available devices (wrist/chest/ankle), posture -----------------

    try:
        data = list(df_epoch1s['wrist_avm'])
        df_out['wrist_avm'] = [np.mean(data[i:i + new_epoch_len]) for
                               i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]
    except (KeyError, TypeError):
        df_out['wrist_avm'] = [None] * len(df_out)

    if cutpoints is None:
        df_out['wrist_intensity'] = [None] * df_out.shape[0]
    if cutpoints is not None:
        df_out['wrist_intensity'] = ['sedentary'] * df_out.shape[0]
        df_out.loc[(df_out['wrist_avm'] >= cp['light']) & (df_out['wrist_avm'] < cp['mod']), 'wrist_intensity'] = 'light'
        df_out.loc[(df_out['wrist_avm'] >= cp['mod']) & (df_out['wrist_avm'] < cp['vig']), 'wrist_intensity'] = 'mod'
        df_out.loc[df_out['wrist_avm'] >= cp['vig'], 'wrist_intensity'] = 'vig'

    try:
        data = list(df_epoch1s['ankle_avm'])
        df_out['ankle_avm'] = [np.mean(data[i:i + new_epoch_len]) for
                               i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]
    except (KeyError, TypeError):
        df_out['ankle_avm'] = [None] * len(df_out)

    data = list(df_epoch1s['chest_avm'])
    try:
        df_out['chest_avm'] = [np.mean(data[i:i + new_epoch_len]) for
                               i in np.arange(0, df_epoch1s.shape[0], new_epoch_len)]
    except (KeyError, TypeError):
        df_out['chest_avm'] = [None] * len(df_out)

    df_out['wrist_active'] = df_out['wrist_avm'] >= avm_thresh

    if 'posture' in df_epoch1s.columns:
        epoch_postures = []
        for row in df_out.itertuples():
            df_post = df_epoch1s.loc[row.Index:row.Index + new_epoch_len]['posture']

            if len(df_post.unique()) == 1:
                epoch_postures.append(df_post.iloc[0])
            else:
                epoch_postures.append(None)

        df_out['posture'] = epoch_postures

        df_out['posture_use'] = epoch_postures
        df_out.loc[df_out['gait_percent'] > 0, 'posture_use'] = 'upright_gait'
        df_out.loc[(~df_out['wrist_active']) & (df_out['posture_use'] == 'upright'), 'posture_use'] = 'upright_inactive'
        df_out.loc[(df_out['wrist_active']) & (df_out['posture_use'] == 'upright') & (df_out['gait_percent'] == 0),
                   'posture_use'] = 'upright_active_nogait'

    df_out.reset_index(drop=True, inplace=True)

    df_out = df_out[['full_id', 'start_time', 'days', 'day_int',
                     'snr', 'snr_quality', 'snr_hr',
                     'posture', 'posture_use',
                     'wrist_avm', 'ankle_avm', 'chest_avm',
                     'wrist_nw_percent', 'ankle_nw_percent', 'chest_nw_percent', 'all_wear',
                     'wrist_intensity', 'wrist_active',
                     'sleep_any', 'sleep_percent', 'sptw_percent',
                     'gait_any', 'gait_percent']]

    return df_out


def bin_avm_data(df_epoch: pd.DataFrame,
                 binsize: int = 10,
                 ignore_nw: bool = True,
                 ignore_sleep: bool = True,
                 x: str = 'wrist',
                 y: str = 'ankle'):
    """ Bins AVM data into bins of specified size and describes SNR data by bin.
        Designed to be used to generate some sort of heatmap-type graph

        Parameters
        ----------
        df_epoch

        binsize
            AVM bin size
        ignore_nw
            If True, ignores nonwear epochs in calculations
        ignore_sleep
            If True, ignores sleep epochs in calculations
        x
            str in ['wrist', 'ankle', 'chest'] used to computer df_desc (corresponds to column values)
        y
            str in ['wrist', 'ankle', 'chest'] used to computer df_desc (corresponds to row values)

        Returns
        -------
        df_values
            dataframe where each row is a bin and each column is the average SNR value when the wrist, ankle,
            or chest AVM, respectively, fall into that bin
        df_desc
            average SNR value when specified location AVM values fall into that xy-coordinate
                e.g.: df_desc[50, 100] is the average SNR when device x's AVM is 100 and y's is 50
    """

    print(f"\n-Binning and analyzing AVM data by {x} x {y}")
    wrist_bin = [int(np.floor(i / binsize) * binsize) if not np.isnan(i) else None for i in df_epoch['wrist_avm']]
    ankle_bin = [int(np.floor(i / binsize) * binsize) if not np.isnan(i) else None for i in df_epoch['ankle_avm']]
    chest_bin = [int(np.floor(i / binsize) * binsize) if not np.isnan(i) else None for i in df_epoch['chest_avm']]

    df_epoch['wrist_avm_bin'] = wrist_bin
    df_epoch['ankle_avm_bin'] = ankle_bin
    df_epoch['chest_avm_bin'] = chest_bin

    if ignore_sleep and 'sleep_any' in df_epoch.columns:
        df = df_epoch.loc[~df_epoch['sleep_any']]
    if ignore_nw and 'chest_nw_percent' in df_epoch.columns:
        df = df_epoch.loc[df_epoch['chest_nw_percent'] == 0]

    avm_range = np.arange(0, max([df_epoch['wrist_avm_bin'].max(), df_epoch['ankle_avm_bin'].max()]) + 1, binsize)
    df_desc = pd.DataFrame(columns=avm_range, index=avm_range)

    min_count = 0
    for avm in avm_range:
        df_use = df_epoch.loc[df_epoch[f'{x}_avm_bin'] == avm]
        df_use = df_use.sort_values(f"{y}_avm_bin", ascending=True)

        if df_use.shape[0] == 0:
            df_desc[avm] = [None] * df_desc.shape[0]

        if df_use.shape[0] >= 1:
            df_group = df_use.groupby(f"{y}_avm_bin")['snr'].describe()
            df_group.loc[df_group['count'] < min_count, 'mean'] = None
            df_desc[avm] = df_group['mean']

    l = ['chest', 'ankle', 'wrist']
    l.remove(x)
    l.remove(y)
    z = l[0]

    df_values = pd.DataFrame({"avm": avm_range,
                              f'{x}_snr': [np.mean(df_desc[col]) for col in avm_range],
                              f'{y}_snr': [np.mean(df_desc.loc[col]) for col in avm_range],
                              f'{z}_snr': [df_epoch.loc[df_epoch[f'{z}_avm_bin'] == avm]['snr'].mean() for
                                           avm in avm_range]})

    df_desc = df_desc.dropna(axis=1, how='all')
    df_desc = df_desc.dropna(axis=0, how='all').sort_index(ascending=False)

    return df_desc, df_values


def crop_df(df: pd.DataFrame,
            start: pd.Timestamp or None = None,
            end: pd.Timestamp or None = None):
    """ Crops dataframe to given start/end times using its "start_time" column.
        If only start/end given, only uses what's given

        Parameters
        ----------
        df
            dataframe to crop. Requires "start_time" column
        start
            start timestamp. rows before this time excluded
        end
            end timestamp. rows after this time excluded

        Returns
        -------
        cropped df
    """

    if start is not None:
        df = df.loc[df['start_time'] >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df['start_time'] <= pd.to_datetime(end)]

    return df


def flag_posture_transitions(df_epoch1s: pd.DataFrame,
                             flag_n_secs: int = 2):
    """ Finds posture changes in 1-second epoch data and flags the surrounding flag_n_sec epochs as 'transition'

        Parameters
        ----------
        df_epoch1s
            1-second epoched data with 'posture' column
        flag_n_secs
            number of seconds around each posture transition that get flagged

        Returns
        -------
        copy of df_epoch1s with 'posture_change' and 'posture_use' columns
            e.g. ['sit', 'sit', 'sit', 'stand', 'stand', 'stand', 'stand', 'stand'] becomes
                 ['sit', 'transition', 'transition', 'transition', 'transition', 'transition', 'stand', 'stand'] when
                 flag_n_secs = 2
    """

    df_epoch = df_epoch1s.copy()

    p = list(df_epoch['posture'])
    posture_change = [p[i] != p[i + 1] for i in range(len(p) - 1)]
    posture_change.insert(0, False)
    df_epoch['posture_change'] = posture_change

    df_epoch['posture_use'] = df_epoch['posture'].copy()
    for row in df_epoch.loc[df_epoch['posture_change']].itertuples():
        df_epoch.loc[row.Index - flag_n_secs:row.Index + flag_n_secs, 'posture_use'] = 'transition'

    return df_epoch


class AllData:

    def __init__(self,
                 full_id: str = "",
                 snr_edf_folder: str = "",
                 edf_folder: str = "",
                 load_ecg: bool = False,
                 df_subj: pd.DataFrame or None = None,
                 med_epoch_len: int = 5,
                 long_epoch_len: int = 900,
                 min_gait_dur: int = 0,
                 min_cadence: int = 0,
                 snr_thresh: tuple or list = (5, 18),
                 sleepbouts_file: str or None = None,
                 sptw_file: str or None = None,
                 gaitbouts_file: str or None = None,
                 df_avm_file: str or None = "",
                 df_snr_filename: str or None = None,
                 posture_file: str or None = None,
                 wrist_nw_file: str or None = None,
                 ankle_nw_file: str or None = None,
                 bittium_nw_file: str = None,
                 pad_nonwear_bouts: int or float = 5,
                 avm_active_thresh: int or float = 15,
                 flag_n_secs_transitions: int = 2,
                 df_1s_filename: str or None = None,
                 start_time: str or pd.Timestamp or None = None,
                 end_time: str or pd.Timestamp or None = None):
        """ Runs most data processing for single participant.

            Parameters
            ----------
            full_id
                participant ID not including collection ID
            snr_edf_folder
                pathway to folder containing SNR timeseries EDF files
            edf_folder
                pathway to folder containing Bittium and Axivity EDF files
            load_ecg
                if True, loads ECG EDF file
            df_subj
                df output from get_collection_details()
            med_epoch_len
                medium epoch length used for primary analysis, in seconds
            long_epoch_len
                long epoch length used for trend over time analysis, in seconds
            min_gait_dur
                minimum gait bout duration to flag gait as occurring
            min_cadence
                minimum gait bout average cadence to flag gait as occurring
            snr_thresh
                list/tuple of SNR thresholds to classify SNR as Q1, Q2, Q3
            sleepbouts_file
                pathway to sleep bouts csv file
            sptw_file
                pathway to Sleep Period Time Windows csv file
            gaitbouts_file
                pathway to gait bouts csv file
            df_avm_file
                pathway to combined wrist/ankle/chest 1-second AVM csv file
            df_snr_filename
                pathway to 1-second average SNR file
            posture_file
                pathway to 1-second posture classification csv file
            wrist_nw_file
                pathway to wrist non-wear bout csv file
            ankle_nw_file
                pathway to ankle non-wear bout csv file
            bittium_nw_file
                pathway to chest non-wear bout csv file
            pad_nonwear_bouts
                number of minutes added to start/end of each nonwear bout for each device to ensure no
                false positive nonwear included in analyses
            avm_active_thresh
                Average Vector Magnitude threshold applied to wrist data to flag the wrist
                as active/inactive (not related to activity intensity)
            flag_n_secs_transitions
                number of seconds flagged as "transitions" on either side of each posture change
                (will get ignored so movement artifact from transitions not assigned to any posture)
            df_1s_filename
                pathway to processed 1-second epoch file. If given, processing will not be re-run. If None,
                processing will be run
            start_time
                If not None, crops df_epoch to after this timestamp
            end_time
                If not None, crops df_epoch to before this timestamp
        """

        # data dictionary of raw Bittium ECG/accel and SNR data
        self.raw = {'ts': [], 'snr': []}
        self.ecg = None

        if load_ecg:
            self.ecg = import_bittium(f"{edf_folder}{full_id}_01_BF36_Chest.edf")

        self.df_epoch = pd.read_csv(df_avm_file) if os.path.exists(df_avm_file) else \
            pd.DataFrame(columns=['start_time', 'days'])

        if df_avm_file is None or not os.path.exists(df_avm_file):
            try:
                self.df_epoch = combine_df_avm(full_id,
                                               wrist_path="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/wrist_avm/",
                                               ankle_path="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/ankle_avm/",
                                               chest_path="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/chest_avm/",
                                               save_dir="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/avm/",
                                               save_file=False)
            except FileNotFoundError:
                print("-combine_df_avm() has failed")

        self.df_epoch['start_time'] = pd.to_datetime(self.df_epoch['start_time'])
        self.df_snr = None

        self.df_epoch_med = pd.DataFrame()
        self.long_epoch_len = long_epoch_len
        self.med_epoch_len = med_epoch_len
        self.min_gait_dur = min_gait_dur
        self.min_cadence = min_cadence
        self.avm_active_thresh = avm_active_thresh

        self.sleepbouts_file = sleepbouts_file
        self.sptw_file = sptw_file
        self.gaitbouts_file = gaitbouts_file
        self.posture_file = posture_file
        self.wrist_nw_file = wrist_nw_file
        self.ankle_nw_file = ankle_nw_file
        self.bittium_nw_file = bittium_nw_file

        self.start_time = start_time
        self.end_time = end_time

        self.snr_thresh = snr_thresh

        # signal quality over time regression values
        self.slope = None
        self.y_int = None

        self.full_id = full_id
        self.snr_edf_folder = snr_edf_folder
        self.edf_folder = edf_folder

        self.df_subj = df_subj

        if df_1s_filename is None or not os.path.exists(df_1s_filename):
            print("\nGenerating data from new...")

            if df_snr_filename is None or not os.path.exists(df_snr_filename):

                self.df_subj, self.raw['snr'], self.raw['ts'] = import_snr_raw(full_id=full_id,
                                                                               check_header=self.df_subj is None,
                                                                               df_subj=df_subj)

            if df_snr_filename is not None and os.path.exists(df_snr_filename):
                print(f"-Importing 1-second SNR average data...")
                self.df_snr = pd.read_csv(df_snr_filename)
                self.df_snr['start_time'] = pd.to_datetime(self.df_snr['start_time'])
                self.df_snr = self.df_snr.loc[(self.df_snr['start_time'] >= self.df_epoch.iloc[0]['start_time']) &
                                              (self.df_snr['start_time'] <= self.df_epoch.iloc[-1]['start_time'])]
                self.df_epoch['snr'] = self.df_snr['snr']

            self.df_subj = self.df_subj.iloc[0]

            self.events = self.create_df_context(min_gait_dur=self.min_gait_dur,
                                                 min_cadence=self.min_cadence,
                                                 pad_nonwear_bouts=pad_nonwear_bouts)

            if self.start_time is not None:
                self.df_epoch = self.df_epoch.loc[self.df_epoch['start_time'] >= self.start_time]
            if self.end_time is not None:
                self.df_epoch = self.df_epoch.loc[self.df_epoch['start_time'] < self.end_time]

            self.df_epoch['full_id'] = [full_id] * self.df_epoch.shape[0]

            self.df_epoch = flag_posture_transitions(self.df_epoch, flag_n_secs=flag_n_secs_transitions)

            self.df_posture_bouts = bout_posture(self.df_epoch)

            self.df_epoch_long = average_snr(snr_signal=self.raw['snr'], timestamps=self.raw['ts'],
                                             sample_rate=self.df_subj['sample_rate'], n_secs=self.long_epoch_len)

        if df_1s_filename is not None:
            print("\nReading 1-second data from csv...")
            self.df_epoch = pd.read_csv(df_1s_filename)
            self.df_epoch['start_time'] = pd.to_datetime(self.df_epoch['start_time'])

            if self.start_time is not None:
                self.df_epoch = self.df_epoch.loc[self.df_epoch['start_time'] >= self.start_time]
            if self.end_time is not None:
                self.df_epoch = self.df_epoch.loc[self.df_epoch['start_time'] <= self.end_time]

            if 'snr' not in self.df_epoch.columns:
                print(f"-Importing 1-second SNR average data...")
                self.df_snr = pd.read_csv(df_snr_filename)
                self.df_snr['start_time'] = pd.to_datetime(self.df_snr['start_time'])
                self.df_snr = self.df_snr.loc[(self.df_snr['start_time'] >= self.df_epoch.iloc[0]['start_time']) &
                                              (self.df_snr['start_time'] <= self.df_epoch.iloc[-1]['start_time'])]
                self.df_epoch['snr'] = self.df_snr['snr']

        self.df_epoch_med = reepoch_data(df_epoch1s=self.df_epoch, new_epoch_len=med_epoch_len,
                                         cutpoints=None, avm_thresh=avm_active_thresh)

        self.df_epoch_long = reepoch_data(df_epoch1s=self.df_epoch, new_epoch_len=long_epoch_len,
                                          cutpoints=None, avm_thresh=avm_active_thresh)

        self.reg = quantify_trend(self.df_epoch_long.loc[self.df_epoch_long['chest_nw_percent'] == 0])

    def plot_epoched_snr(self):

        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(self.df_epoch_med['days'], self.df_epoch_med['snr'], color='black', label=f"{self.med_epoch_len}-sec")
        ax.plot(self.df_epoch_long['days'], self.df_epoch_long['snr'], color='dodgerblue',
                label=f"{self.long_epoch_len}-sec")

        ax.plot(self.df_epoch['days'], [i * self.reg['slope'] + self.reg['int'] for i in self.df_epoch['days']],
                color='red', linestyle='dashed',
                label=f"dB = {self.reg['slope']:.3f}days + {self.reg['int']:.3f}\n(p={self.reg['p']:.3f})")
        ax.legend()
        ax.set_ylabel("dB")
        ax.set_xlabel("Days into collection")

        ax.axhspan(xmin=0, xmax=self.df_epoch_med['days'].max(),
                   ymin=self.df_epoch_med['snr'].min(), ymax=self.snr_thresh[0],
                   color='red', alpha=.2)
        ax.axhspan(xmin=0, xmax=self.df_epoch_med['days'].max(),
                   ymin=self.snr_thresh[0], ymax=self.snr_thresh[1],
                   color='dodgerblue', alpha=.2)
        ax.axhspan(xmin=0, xmax=self.df_epoch_med['days'].max(),
                   ymin=self.snr_thresh[1], ymax=self.df_epoch_med['snr'].max(),
                   color='limegreen', alpha=.2)

        ax.grid()
        plt.tight_layout()

    def create_df_context(self,
                          min_gait_dur: int = 0,
                          min_cadence: int or float = 0,
                          pad_nonwear_bouts: int or float = 5):

        df_gait = pd.DataFrame()
        df_sleep = pd.DataFrame()
        sptw = pd.DataFrame()
        wrist_nw = pd.DataFrame()
        ankle_nw = pd.DataFrame()
        chest_nw = pd.DataFrame()
        df_posture = pd.DataFrame()

        start_time = self.df_epoch.iloc[0]['start_time']
        end_time = self.df_epoch.iloc[-1]['start_time']
        epoch_len = int((self.df_epoch.iloc[1]['start_time'] - start_time).total_seconds())

        if 'days' not in self.df_epoch.columns:
            self.df_epoch['days'] = [(row.start_time - start_time).total_seconds() / 86400 for
                                     row in self.df_epoch.itertuples()]

        print("\nChecking tabular data files to provide context...")

        epoch_samples = int(epoch_len * self.df_subj['sample_rate'])

        if 'snr' not in self.df_epoch.columns:
            self.df_epoch['snr'] = [np.mean(self.raw['snr'][i:i+epoch_samples]) for
                                    i in np.arange(0, len(self.raw['snr']), epoch_samples)]

        self.df_epoch['sleep_mask'] = [None] * self.df_epoch.shape[0]  # binary sleep bouts: sleep (1), wake (0)
        self.df_epoch['sptw_mask'] = [None] * self.df_epoch.shape[0]  # binary sleep period time window: sleep (1), wake (0)

        self.df_epoch['gait_mask'] = [None] * self.df_epoch.shape[0]  # binary gait: gait (1), no gait (0)

        self.df_epoch['posture'] = [None] * self.df_epoch.shape[0]  # postures as str
        self.df_epoch['upright_mask'] = [None] * self.df_epoch.shape[0]  # upright posture: upright (1), other (0)
        self.df_epoch['supine_mask'] = [None] * self.df_epoch.shape[0]  # supine posture: supine (1), other (0)
        self.df_epoch['prone_mask'] = [None] * self.df_epoch.shape[0]  # prone posture: prone (1), other (0)
        self.df_epoch['side_mask'] = [None] * self.df_epoch.shape[0]  # side lying posture: side lying (1), other (0)

        self.df_epoch['wrist_nw_mask'] = [None] * self.df_epoch.shape[0]
        self.df_epoch['ankle_nw_mask'] = [None] * self.df_epoch.shape[0]
        self.df_epoch['chest_nw_mask'] = [None] * self.df_epoch.shape[0]

        if self.sleepbouts_file is not None:
            print("-Importing sleep data...")
            df_sleep = pd.read_csv(self.sleepbouts_file)
            df_sleep['start_time'] = pd.to_datetime(df_sleep['start_time'])
            df_sleep['end_time'] = pd.to_datetime(df_sleep['end_time'])

            # sets bouts that end after end_time to end_time
            df_sleep.loc[df_sleep['end_time'] >= end_time] = end_time

            df_sleep = df_sleep.loc[(df_sleep['start_time'] >= start_time) &
                                    (df_sleep['start_time'] < end_time)]

            self.df_epoch['sleep_mask'] = create_mask(start_time=start_time,
                                                      n_rows=self.df_epoch.shape[0],
                                                      df=df_sleep,
                                                      epoch_len=epoch_len)

        if self.sptw_file is not None:
            print("-Importing sleep period time window data...")
            sptw = pd.read_csv(self.sptw_file)
            sptw['start_time'] = pd.to_datetime(sptw['start_time'])
            sptw['end_time'] = pd.to_datetime(sptw['end_time'])

            # sets bouts that end after end_time to end_time
            sptw.loc[sptw['end_time'] >= end_time] = end_time

            sptw = sptw.loc[(sptw['start_time'] >= start_time) & (sptw['start_time'] < end_time)]

            self.df_epoch['sptw_mask'] = create_mask(start_time=start_time,
                                                     n_rows=self.df_epoch.shape[0],
                                                     df=sptw,
                                                     epoch_len=epoch_len)

        if self.gaitbouts_file is not None:
            print("-Importing gait data...")

            df_gait = pd.read_csv(self.gaitbouts_file)
            df_gait['start_time'] = pd.to_datetime(df_gait['start_timestamp'] if
                                                   'start_timestamp' in df_gait.columns else df_gait['start_time'])
            df_gait['end_time'] = pd.to_datetime(df_gait['end_timestamp'] if
                                                 'end_timestamp' in df_gait.columns else df_gait['end_time'])
            df_gait = df_gait[[i for i in df_gait.columns if i not in ['start_timestamp', 'end_timestamp']]]
            df_gait['duration'] = [(row.end_time - row.start_time).total_seconds() for row in df_gait.itertuples()]
            df_gait['cadence'] = df_gait['step_count'] / df_gait['duration'] * 60

            # sets bouts that end after end_time to end_time
            df_gait.loc[df_gait['end_time'] >= end_time] = end_time

            df_gait = df_gait.loc[(df_gait['start_time'] >= start_time) & (df_gait['start_time'] < end_time)]

            if min_gait_dur > 0:
                print(f"    -Omitting bouts shorter than {min_gait_dur} seconds")
                df_gait = df_gait.loc[df_gait['duration'] >= min_gait_dur]
                df_gait.reset_index(drop=True, inplace=True)

            if min_cadence > 0:
                print(f"    -Omitting bouts with average cadence below {min_cadence} steps/min")
                df_gait = df_gait.loc[df_gait['cadence'] >= min_cadence]
                df_gait.reset_index(drop=True, inplace=True)

            self.df_epoch['gait_mask'] = create_mask(start_time=start_time,
                                                     n_rows=self.df_epoch.shape[0],
                                                     df=df_gait,
                                                     epoch_len=epoch_len)

        if self.posture_file is not None:
            print("-Importing posture data...")
            df_posture = import_posture(self.posture_file)
            df_posture['code'] = df_posture['posture'].replace({"upright": 1, 'supine': 2, 'prone': 3,
                                                                'rightside': 4, 'leftside': 5,
                                                                'reclined': 6, 'other': 7})
            code_dict = {1: 'upright', 2: 'supine', 3: 'prone', 4: 'rightside',
                         5: 'leftside', 6: 'reclined', 7: 'other', 0.0: 'other'}

            self.df_epoch['upright_mask'] = create_mask(start_time=self.df_epoch['start_time'].iloc[0],
                                                        n_rows=self.df_epoch.shape[0],
                                                        df=df_posture.loc[df_posture['posture'] == 'upright'],
                                                        epoch_len=epoch_len)

            self.df_epoch['supine_mask'] = create_mask(start_time=self.df_epoch['start_time'].iloc[0],
                                                       n_rows=self.df_epoch.shape[0],
                                                       df=df_posture.loc[df_posture['posture'] == 'supine'],
                                                       epoch_len=epoch_len)

            self.df_epoch['prone_mask'] = create_mask(start_time=self.df_epoch['start_time'].iloc[0],
                                                      n_rows=self.df_epoch.shape[0],
                                                      df=df_posture.loc[df_posture['posture'] == 'prone'],
                                                      epoch_len=epoch_len)

            self.df_epoch['side_mask'] = create_mask(start_time=self.df_epoch['start_time'].iloc[0],
                                                     n_rows=self.df_epoch.shape[0],
                                                     df=df_posture.loc[df_posture['posture'].isin(['leftside', 'rightside'])],
                                                     epoch_len=epoch_len)

            # postures as str. this is awful code. don't judge me.
            mask = np.zeros(self.df_epoch.shape[0])
            start_time = self.df_epoch['start_time'].iloc[0]

            for row in df_posture.itertuples():
                try:
                    start_i = int((row.start_time - start_time).total_seconds() / epoch_len)
                    end_i = int((row.end_time - start_time).total_seconds() / epoch_len)
                    mask[start_i:end_i] = row.code

                except AttributeError:
                    start_i = int((row.start_time - start_time).total_seconds() / epoch_len)
                    mask[start_i] = row.code

            self.df_epoch['posture'] = [code_dict[i] for i in mask]
            self.df_epoch['posture'].replace({"rightside": 'sidelying', 'leftside': 'sidelying', 'reclined': 'supine'},
                                             inplace=True)

        if self.bittium_nw_file is not None:
            print("-Importing Bittium non-wear data...")
            chest_nw = pd.read_csv(self.bittium_nw_file)
            chest_nw['start_time'] = pd.to_datetime(chest_nw['start_time'])
            chest_nw['end_time'] = pd.to_datetime(chest_nw['end_time'])

            if pad_nonwear_bouts > 0:
                print(f"    -Padding nonwear bouts by {pad_nonwear_bouts} minutes")
                chest_nw['start_time'] -= timedelta(minutes=pad_nonwear_bouts)
                chest_nw['end_time'] += timedelta(minutes=pad_nonwear_bouts)

            # sets bouts that end after end_time to end_time
            chest_nw.loc[chest_nw['end_time'] >= end_time] = end_time

            chest_nw = chest_nw.loc[(chest_nw['start_time'] >= start_time) & (chest_nw['start_time'] < end_time)]

            self.df_epoch['chest_nw_mask'] = create_mask(start_time=start_time,
                                                         n_rows=self.df_epoch.shape[0],
                                                         df=chest_nw,
                                                         epoch_len=epoch_len)

        if self.ankle_nw_file is not None:
            print("-Importing ankle non-wear data...")
            ankle_nw = pd.read_csv(self.ankle_nw_file)
            ankle_nw['start_time'] = pd.to_datetime(ankle_nw['start_time'])
            ankle_nw['end_time'] = pd.to_datetime(ankle_nw['end_time'])

            if pad_nonwear_bouts > 0:
                print(f"    -Padding ankle nonwear bouts by {pad_nonwear_bouts} minutes")
                ankle_nw['start_time'] -= timedelta(minutes=pad_nonwear_bouts)
                ankle_nw['end_time'] += timedelta(minutes=pad_nonwear_bouts)

            # sets bouts that end after end_time to end_time
            ankle_nw.loc[ankle_nw['end_time'] >= end_time] = end_time

            ankle_nw = ankle_nw.loc[(ankle_nw['start_time'] >= start_time) & (ankle_nw['start_time'] < end_time)]

            self.df_epoch['ankle_nw_mask'] = create_mask(start_time=start_time,
                                                         n_rows=self.df_epoch.shape[0],
                                                         df=ankle_nw,
                                                         epoch_len=epoch_len)

        if self.wrist_nw_file is not None:
            print("-Importing wrist non-wear data...")
            wrist_nw = pd.read_csv(self.wrist_nw_file)
            wrist_nw['start_time'] = pd.to_datetime(wrist_nw['start_time'])
            wrist_nw['end_time'] = pd.to_datetime(wrist_nw['end_time'])

            if pad_nonwear_bouts > 0:
                print(f"    -Padding wrist nonwear bouts by {pad_nonwear_bouts} minutes")
                wrist_nw['start_time'] -= timedelta(minutes=pad_nonwear_bouts)
                wrist_nw['end_time'] += timedelta(minutes=pad_nonwear_bouts)

            # sets bouts that end after end_time to end_time
            wrist_nw.loc[wrist_nw['end_time'] >= end_time] = end_time

            wrist_nw = wrist_nw.loc[(wrist_nw['start_time'] >= start_time) & (wrist_nw['start_time'] < end_time)]

            self.df_epoch['wrist_nw_mask'] = create_mask(start_time=start_time,
                                                         n_rows=self.df_epoch.shape[0],
                                                         df=wrist_nw,
                                                         epoch_len=epoch_len)

        self.df_epoch['all_wear'] = [row.wrist_nw_mask + row.ankle_nw_mask + row.chest_nw_mask == 0 for
                                     row in self.df_epoch.itertuples()]

        print("\nComplete.")

        out_dict = {'gait': df_gait,
                    'sleep': df_sleep,
                    'sptw': sptw,
                    'posture': df_posture,
                    'wrist_nw': wrist_nw,
                    'ankle_nw': ankle_nw,
                    'chest_nw': chest_nw}

        return out_dict

    def plot_context(self, colnames: list or tuple = ()):

        n_plots = len(colnames) + 1

        gridspec = [1]
        for i in range(len(colnames)):
            gridspec.append(.5)

        fig, ax = plt.subplots(n_plots, sharex='col', figsize=(12, 8), gridspec_kw={'height_ratios': gridspec})

        ax[0].plot(self.df_epoch['start_time'], self.df_epoch['snr'], color='black', label='1-sec')
        ax[0].plot(self.df_epoch_long['start_time'], self.df_epoch_long['snr'],
                   color='dodgerblue', label=f'{self.long_epoch_len}-sec')
        ax[0].set_ylabel("dB")
        ax[0].grid()
        ax[0].legend()

        for i, col in enumerate(colnames):
            ax[i + 1].plot(self.df_epoch['start_time'], self.df_epoch[col], color='black')
            ax[i + 1].set_ylabel(col)
            ax[i + 1].set_yticks([0, 1])

        plt.tight_layout()

        return fig
