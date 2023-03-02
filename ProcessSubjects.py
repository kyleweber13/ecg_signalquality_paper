from tqdm import tqdm
import os
import pandas as pd
import warnings
import nimbalwear
warnings.filterwarnings('ignore')
from ProcessData import reepoch_data, convert_time_columns, flag_posture_transitions, crop_df, crop_df_bouts
from nimbalwear.activity import activity_wrist_avm


def file_check(df_demos,
               df_timestamps=None,
               edf_folders="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
               snr_edf_folder="W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/",
               snr1s_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/snr_1s/",
               epoch1s_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch1s/",
               epochmed_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch_med/",
               med_epoch_len=5,
               epochlong_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch_long/",
               long_epoch_len=900,
               posture_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/chest_posture/",
               sleep_folder="W:/NiMBaLWEAR/OND09/analytics/sleep/bouts/",
               gait_folder="W:/NiMBaLWEAR/OND09/analytics/gait/bouts/",
               avm_root_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/",
               # vert_nonwear_folder="W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/",
               vert_nonwear_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/vert_nonwear/",
               weber_nonwear_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/weber_bittium_nonwear/"):

    df_out = pd.DataFrame({'full_id': df_demos['full_id']})

    wrist_edfs = []
    ankle_edfs = []
    bf_edfs = []

    for row in df_demos.itertuples():
        wrist = "{}{}_01_AXV6_{}Wrist.edf"
        ankle = "{}{}_01_AXV6_{}Ankle.edf"
        bf = "{}{}_01_BF36_Chest.edf"

        if os.path.exists(wrist.format(edf_folders, row.full_id, row.wrist_side)):
            wrist_edfs.append(wrist.format(edf_folders, row.full_id, row.wrist_side))
        if not os.path.exists(wrist.format(edf_folders, row.full_id, row.wrist_side)):
            wrist_edfs.append(None)

        if os.path.exists(ankle.format(edf_folders, row.full_id, row.ankle_side)):
            ankle_edfs.append(ankle.format(edf_folders, row.full_id, row.ankle_side))
        if not os.path.exists(ankle.format(edf_folders, row.full_id, row.ankle_side)):
            ankle_edfs.append(None)

        if os.path.exists(bf.format(edf_folders, row.full_id)):
            bf_edfs.append(bf.format(edf_folders, row.full_id))
        if not os.path.exists(bf.format(edf_folders, row.full_id)):
            bf_edfs.append(None)

    df_out['wrist_edf_file'] = wrist_edfs
    df_out['ankle_edf_file'] = ankle_edfs
    df_out['chest_edf_file'] = bf_edfs

    df_out['start_time'] = [df_timestamps.loc[df_timestamps['full_id'] == row.full_id].iloc[0]['use_start'] for row in df_out.itertuples()]
    df_out['end_time'] = [df_timestamps.loc[df_timestamps['full_id'] == row.full_id].iloc[0]['use_end'] for row in df_out.itertuples()]
    df_out['effective_dur_h'] = [(row.end_time - row.start_time).total_seconds()/3600 for row in df_out.itertuples()]

    df_out['snr_edf'] = [os.path.exists(f"{snr_edf_folder}{full_id}_01_snr.edf") for full_id in df_demos['full_id']]
    df_out['snr_edf_file'] = [f"{snr_edf_folder}{full_id}_01_snr.edf" for full_id in df_demos['full_id']]

    df_out['snr1s'] = [os.path.exists(f"{snr1s_folder}{full_id}_snr1s.csv") for full_id in df_demos['full_id']]
    df_out['snr1s_file'] = [f"{snr1s_folder}{full_id}_snr1s.csv" for full_id in df_demos['full_id']]

    df_out['epoch1s'] = [os.path.exists(f"{epoch1s_folder}{full_id}_epoch1s.csv") for full_id in df_demos['full_id']]
    df_out['epoch1s_file'] = [f"{epoch1s_folder}{full_id}_epoch1s.csv" for full_id in df_demos['full_id']]

    df_out['epoch_med'] = [os.path.exists(f"{epochmed_folder}{full_id}_epoch{med_epoch_len}.csv") for full_id in df_demos['full_id']]
    df_out['epoch_med_file'] = [f"{epochmed_folder}{full_id}_epoch{med_epoch_len}.csv" for full_id in df_demos['full_id']]

    df_out['epoch_long'] = [os.path.exists(f"{epochlong_folder}{full_id}_epoch{long_epoch_len}.csv") for full_id in df_demos['full_id']]
    df_out['epoch_long_file'] = [f"{epochlong_folder}{full_id}_epoch{long_epoch_len}.csv" for full_id in df_demos['full_id']]

    df_out['posture'] = [os.path.exists(f"{posture_folder}{full_id}_01_posture.csv") for full_id in df_demos['full_id']]
    df_out['posture_file'] = [f"{posture_folder}{full_id}_01_posture.csv" for full_id in df_demos['full_id']]

    df_out['sleep'] = [os.path.exists(f"{sleep_folder}{full_id}_01_SLEEP_BOUTS.csv") for full_id in df_demos['full_id']]
    df_out['sleep_file'] = [f"{sleep_folder}{full_id}_01_SLEEP_BOUTS.csv" for full_id in df_demos['full_id']]

    df_out['gait'] = [os.path.exists(f"{gait_folder}{full_id}_01_GAIT_BOUTS.csv") for full_id in df_demos['full_id']]
    df_out['gait_file'] = [f"{gait_folder}{full_id}_01_GAIT_BOUTS.csv" for full_id in df_demos['full_id']]

    df_out['avm_combined'] = [os.path.exists(f"{avm_root_folder}avm/{full_id}_01_avm.csv") for full_id in df_demos['full_id']]
    df_out['avm_combined_file'] = [f"{avm_root_folder}avm/{full_id}_01_avm.csv" for full_id in df_demos['full_id']]

    df_out['wrist_avm'] = [os.path.exists(f"{avm_root_folder}wrist_avm/{full_id}_01_Wrist_AVM.csv") for full_id in df_demos['full_id']]
    df_out['wrist_avm_file'] = [f"{avm_root_folder}wrist_avm/{full_id}_01_Wrist_AVM.csv" for full_id in df_demos['full_id']]

    df_out['ankle_avm'] = [os.path.exists(f"{avm_root_folder}ankle_avm/{full_id}_01_Ankle_AVM.csv") for full_id in df_demos['full_id']]
    df_out['ankle_avm_file'] = [f"{avm_root_folder}ankle_avm/{full_id}_01_Ankle_AVM.csv" for full_id in df_demos['full_id']]

    df_out['chest_avm'] = [os.path.exists(f"{avm_root_folder}chest_avm/{full_id}_01_Chest_AVM.csv") for full_id in df_demos['full_id']]
    df_out['chest_avm_file'] = [f"{avm_root_folder}chest_avm/{full_id}_01_Chest_AVM.csv" for full_id in df_demos['full_id']]

    df_out['wrist_nw'] = [os.path.exists(f"{vert_nonwear_folder}{full_id}_01_AXV6_RWrist_NONWEAR.csv") or
                          os.path.exists(f"{vert_nonwear_folder}{full_id}_01_AXV6_LWrist_NONWEAR.csv") for full_id in df_demos['full_id']]

    df_out['ankle_nw'] = [os.path.exists(f"{vert_nonwear_folder}{full_id}_01_AXV6_RAnkle_NONWEAR.csv") or
                          os.path.exists(f"{vert_nonwear_folder}{full_id}_01_AXV6_LAnkle_NONWEAR.csv") for full_id in df_demos['full_id']]
    df_out['chest_nw'] = [os.path.exists(f"{vert_nonwear_folder}{full_id}_01_BF36_Chest_NONWEAR.csv") for full_id in df_demos['full_id']]
    df_out['chest_weber_nw'] = [os.path.exists(f"{weber_nonwear_folder}{full_id}_01_BF36_Chest_NONWEAR.csv") for full_id in df_demos['full_id']]

    wrist_nw = []
    ankle_nw = []
    chest_nw = []
    chest_nw_weber = []
    for full_id in df_demos['full_id']:
        added_wrist = False
        added_ankle = False

        if os.path.exists(f"{vert_nonwear_folder}{full_id}_01_AXV6_RWrist_NONWEAR.csv") and not added_wrist:
            wrist_nw.append(f"{vert_nonwear_folder}{full_id}_01_AXV6_RWrist_NONWEAR.csv")
            added_wrist = True

        if os.path.exists(f"{vert_nonwear_folder}{full_id}_01_AXV6_LWrist_NONWEAR.csv") and not added_wrist:
            wrist_nw.append(f"{vert_nonwear_folder}{full_id}_01_AXV6_LWrist_NONWEAR.csv")
            added_wrist = True

        if not added_wrist:
            wrist_nw.append(None)

        if os.path.exists(f"{vert_nonwear_folder}{full_id}_01_AXV6_RAnkle_NONWEAR.csv") and not added_ankle:
            ankle_nw.append(f"{vert_nonwear_folder}{full_id}_01_AXV6_RAnkle_NONWEAR.csv")
            added_ankle = True

        if os.path.exists(f"{vert_nonwear_folder}{full_id}_01_AXV6_LAnkle_NONWEAR.csv") and not added_ankle:
            ankle_nw.append(f"{vert_nonwear_folder}{full_id}_01_AXV6_LAnkle_NONWEAR.csv")
            added_ankle = True

        if not added_ankle:
            ankle_nw.append(None)

        if os.path.exists(f"{vert_nonwear_folder}{full_id}_01_BF36_Chest_NONWEAR.csv"):
            chest_nw.append(f"{vert_nonwear_folder}{full_id}_01_BF36_Chest_NONWEAR.csv")
        if not os.path.exists(f"{vert_nonwear_folder}{full_id}_01_BF36_Chest_NONWEAR.csv"):
            chest_nw.append(None)

        if os.path.exists(f"{weber_nonwear_folder}{full_id}_01_BF36_Chest_NONWEAR.csv"):
            chest_nw_weber.append(f"{weber_nonwear_folder}{full_id}_01_BF36_Chest_NONWEAR.csv")
        if not os.path.exists(f"{weber_nonwear_folder}{full_id}_01_BF36_Chest_NONWEAR.csv"):
            chest_nw_weber.append(None)

    df_out['wrist_nw_file'] = wrist_nw
    df_out['ankle_nw_file'] = ankle_nw
    df_out['chest_nw_file'] = chest_nw
    df_out['chest_weber_nw_file'] = chest_nw_weber

    df_out['individual_avm'] = df_out['wrist_nw'] & df_out['ankle_nw'] & df_out['chest_nw']

    df_out['complete'] = [(row.snr_edf + row.snr1s + row.epoch1s + row.epoch_med + row.epoch_long +
                           row.posture + row.sleep + row.gait + row.avm_combined +
                           row.wrist_nw + row.ankle_nw + row.chest_nw + row.chest_weber_nw) == 13 for row in df_out.itertuples()]

    bool_cols = ['snr_edf', 'snr1s', 'epoch1s', 'epoch_med', 'epoch_long', 'posture',
                 'sleep', 'gait', 'avm_combined', 'wrist_nw', 'ankle_nw', 'chest_nw']
    df_out['missing'] = [[column for column in bool_cols if
                          df_out.loc[df_out['full_id'] == row.full_id][column].iloc[0] is False] for
                         row in df_out.itertuples()]

    for column in ['snr_edf', 'snr1s', 'epoch1s', 'epoch_med', 'epoch_long', 'posture',
                   'sleep', 'gait',  'avm_combined', 'wrist_avm', 'ankle_avm',
                   'chest_avm', 'wrist_nw', 'ankle_nw', 'chest_nw', 'chest_weber_nw', 'complete']:
        print(f"-{column}: {df_out[column].sum()}/{df_out.shape[0]}")

    df_out.index = df_out['full_id']

    return df_out


def process_posture(subjs, save_dir="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/chest_posture/"):

    import nimbalwear
    from TrunkPosture import calculate_chest_posture, bout_posture

    failed = []
    success = []

    for subj in subjs:

        full_id = f'{subj}_01'

        try:
            ecg = nimbalwear.Device()
            ecg.import_edf(f"W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/{full_id}_BF36_Chest.edf")

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
            success.append(subj)

        except:
            failed.append(subj)


def process_subjs(df_files: pd.DataFrame,
                  epoch_med: int = 5,
                  epoch_long: int = 900,
                  avm_thresh: int or float = 15,
                  snr_thresh: list or tuple = (5, 18),
                  nw_pad_mins: int or float = 5,
                  incl_ignored_weber_nw: bool = False,
                  return_data: bool = True):

    subjs_dict = {}

    df_status = pd.DataFrame(columns=['full_id', 'snr1s_file', 'epoch1s_file', 'epoch_med_file',
                                      'epoch_long_file',  'posture_file', 'avm_combined_file',
                                      'wrist_avm_file', 'ankle_avm_file', 'chest_avm_file'])
    df_status['full_id'] = df_files['full_id']

    for col in df_status.columns:
        df_status[col] = [None] * df_status.shape[0]

    for i in tqdm(range(df_files.shape[0])):
        row = df_files.iloc[i]

        subj_dict = {'df1s': None, 'epoch_med': None, 'epoch_long': None,
                     'snr': None, 'avm': None,
                     'posture': None, 'posture_bouts': None, 'sleep': None, 'gait': None,
                     'wrist_nw': None, 'ankle_nw': None, 'chest_nw': None}

        chest = None

        print(f"=============== {row.full_id} ===============")
        if row.epoch1s:

            df_status.loc[df_status['full_id'] == row.full_id, 'epoch1s_file'] = 'exists'

            if row.epoch_med and row.epoch_long:
                print("Participant's data is fully processed. NEXT!")

            subj_dict['df1s'] = pd.read_csv(row.epoch1s_file)
            subj_dict['df1s'] = convert_time_columns(subj_dict['df1s'])

            subj_dict['df1s'] = crop_df(df=subj_dict['df1s'], start=row.start_time, end=row.end_time)

            if 'posture_use' not in subj_dict['df1s'].columns:
                subj_dict["df1s"] = flag_posture_transitions(df_epoch1s=subj_dict['df1s'],
                                                             flag_n_secs=2)

                subj_dict['df1s'].to_csv(row.epoch1s_file, index=False)

            if not row.epoch_med:
                try:
                    print("Processing medium epoch length data")
                    subj_dict['epoch_med'] = reepoch_data(full_id=row.full_id,
                                                          df_epoch1s=subj_dict['df1s'],
                                                          new_epoch_len=epoch_med,
                                                          cutpoints=None,
                                                          snr_thresh=snr_thresh,
                                                          avm_thresh=avm_thresh,
                                                          quiet=True)

                    subj_dict['epoch_med'].to_csv(row.epoch_med_file, index=False)
                    df_status.loc[df_status['full_id'] == row.full_id, 'epoch_med_file'] = 'success'

                except:
                    df_status.loc[df_status['full_id'] == row.full_id, 'epoch_med_file'] = 'failed'

            if not row.epoch_long:
                try:
                    print("Processing long epoch length data")
                    subj_dict['epoch_long'] = reepoch_data(full_id=row.full_id,
                                                           df_epoch1s=subj_dict['df1s'],
                                                           new_epoch_len=epoch_long,
                                                           cutpoints=None,
                                                           snr_thresh=snr_thresh,
                                                           avm_thresh=avm_thresh,
                                                           quiet=True)

                    subj_dict['epoch_long'].to_csv(row.epoch_long_file, index=False)

                    df_status.loc[df_status['full_id'] == row.full_id, 'epoch_long_file'] = 'success'
                    row.epoch_long = True

                except:
                    df_status.loc[df_status['full_id'] == row.full_id, 'epoch_long_file'] = 'failed'

        if not row.epoch1s:

            # SNR ----------------------------------------------
            if not row.snr1s:
                print("No SNR file")

            if row.snr1s:
                subj_dict['snr'] = pd.read_csv(row.snr1s_file)
                subj_dict['snr'] = convert_time_columns(subj_dict['snr'])
                subj_dict['snr'] = crop_df(df=subj_dict['snr'], start=row.start_time, end=row.end_time)

                df_status.loc[df_status['full_id'] == row.full_id, 'snr_file'] = 'exists'

            # Sleep -------------------------------------------
            if row.sleep:
                subj_dict['sleep'] = pd.read_csv(row.sleep_file)
                subj_dict['sleep'] = convert_time_columns(subj_dict['sleep'])
                subj_dict['sleep'] = crop_df_bouts(df=subj_dict['sleep'], start=row.start_time, end=row.end_time)

            if not row.sleep:
                print("No sleep data")

            # Gait --------------------------------------------
            if row.gait:
                subj_dict['gait'] = pd.read_csv(row.gait_file)
                subj_dict['gait'] = convert_time_columns(subj_dict['gait'])
                subj_dict['gait'].columns = [i if i != 'start_timestamp' else 'start_time'for i in subj_dict['gait'].columns]
                subj_dict['gait'].columns = [i if i != 'end_timestamp' else 'end_time'for i in subj_dict['gait'].columns]
                subj_dict['gait'] = crop_df_bouts(df=subj_dict['gait'], start=row.start_time, end=row.end_time)

            if not row.gait:
                print("No gait data")

            # Posture -----------------------------------------
            if not row.posture:
                try:
                    from TrunkPosture import calculate_chest_posture, bout_posture
                    print("Running posture processing ----------")
                    chest = nimbalwear.Device()
                    chest.import_edf(f"W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/{row.full_id}_BF36_Chest.edf",
                                     quiet=False)

                    # 1-second posture classification
                    subj_dict['posture'] = calculate_chest_posture(chest_x=chest.signals[chest.get_signal_index('Accelerometer x')],
                                                                   chest_y=chest.signals[chest.get_signal_index('Accelerometer y')],
                                                                   chest_z=chest.signals[chest.get_signal_index('Accelerometer z')],
                                                                   chest_freq=chest.signal_headers[chest.get_signal_index('Accelerometer x')]['sample_rate'],
                                                                   chest_start=chest.header['start_datetime'])
                    subj_dict['posture'] = crop_df(df=subj_dict['posture'], start=row.start_time, end=row.end_time)

                    # start/end times for each posture bout
                    subj_dict['posture_bouts'] = bout_posture(subj_dict['posture'])

                    subj_dict['posture'].to_csv(row.posture_file, index=False)
                    subj_dict['posture_bouts'].to_csv(row.posture_file.split("_posture.csv")[0] + "_posture_bouts.csv", index=False)
                    df_status.loc[df_status['full_id'] == row.full_id, 'posture_file'] = 'success'

                    # memory management unless needs to read in later
                    if row.avm_combined:
                        del chest

                except:
                    df_status.loc[df_status['full_id'] == row.full_id, 'posture_file'] = 'failed'

            if row.posture:
                subj_dict['posture'] = pd.read_csv(row.posture_file)
                subj_dict['posture'] = convert_time_columns(subj_dict['posture'])
                subj_dict['posture'] = crop_df(df=subj_dict['posture'], start=row.start_time, end=row.end_time)

                df_status.loc[df_status['full_id'] == row.full_id, 'posture_file'] = 'exists'

            # Combined AVM -----------------------------------
            if row.avm_combined:
                subj_dict['avm'] = pd.read_csv(row.avm_combined_file)
                subj_dict['avm'] = convert_time_columns(subj_dict['avm'])
                subj_dict['avm'] = crop_df(df=subj_dict['avm'], start=row.start_time, end=row.end_time)

                df_status.loc[df_status['full_id'] == row.full_id, 'avm_combined_file'] = 'exists'
                df_status.loc[df_status['full_id'] == row.full_id, 'wrist_avm_file'] = 'not_needed'
                df_status.loc[df_status['full_id'] == row.full_id, 'ankle_avm_file'] = 'not_needed'
                df_status.loc[df_status['full_id'] == row.full_id, 'chest_avm_file'] = 'not_needed'

            if not row.avm_combined:

                if not row.wrist_avm:
                    try:
                        print("Running wrist AVM processing ----------")
                        wrist = nimbalwear.Device()
                        wrist.import_edf(row.wrist_edf_file, quiet=False)

                        subj_dict['wrist_avm'] = activity_wrist_avm(x=wrist.signals[wrist.get_signal_index('Accelerometer x')],
                                                                    y=wrist.signals[wrist.get_signal_index('Accelerometer y')],
                                                                    z=wrist.signals[wrist.get_signal_index('Accelerometer z')],
                                                                    epoch_length=1,
                                                                    sample_rate=wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate'],
                                                                    start_datetime=wrist.header['start_datetime'])[0]
                        subj_dict['wrist_avm'] = crop_df(df=subj_dict['wrist_avm'], start=row.start_time, end=row.end_time)

                        subj_dict['wrist_avm'].to_csv(row.wrist_avm_file, index=False)

                        del wrist

                        df_status.loc[df_status['full_id'] == row.full_id, 'wrist_avm_file'] = 'failed'

                    except:
                        df_status.loc[df_status['full_id'] == row.full_id, 'wrist_avm_file'] = 'failed'

                if not row.ankle_avm:
                    try:
                        print("Running ankle AVM processing ----------")
                        ankle = nimbalwear.Device()
                        ankle.import_edf(row.ankle_edf_file, quiet=False)

                        start_key = 'startdate' if 'startdate' in ankle.header.keys() else 'start_datetime'
                        ankle_avm = activity_wrist_avm(x=ankle.signals[ankle.get_signal_index('Accelerometer x')],
                                                       y=ankle.signals[ankle.get_signal_index('Accelerometer y')],
                                                       z=ankle.signals[ankle.get_signal_index('Accelerometer z')],
                                                       epoch_length=1,
                                                       sample_rate=
                                                       ankle.signal_headers[ankle.get_signal_index('Accelerometer x')][
                                                           'sample_rate'],
                                                       start_datetime=ankle.header[start_key])[0]
                        subj_dict['ankle_avm'] = crop_df(df=subj_dict['ankle_avm'], start=row.start_time, end=row.end_time)

                        ankle_avm.to_csv(row.ankle_avm_file, index=False)

                        del ankle
                        df_status.loc[df_status['full_id'] == row.full_id, 'ankle_avm_file'] = 'success'

                    except:
                        df_status.loc[df_status['full_id'] == row.full_id, 'ankle_avm_file'] = 'failed'

                if not row.chest_avm:
                    try:
                        print("Running chest AVM processing ----------")
                        if chest is None:
                            chest = nimbalwear.Device()
                            chest.import_edf(row.chest_edf_file, quiet=False)

                        start_key = 'startdate' if 'startdate' in chest.header.keys() else 'start_datetime'
                        chest_avm = activity_wrist_avm(x=chest.signals[chest.get_signal_index('Accelerometer x')],
                                                       y=chest.signals[chest.get_signal_index('Accelerometer y')],
                                                       z=chest.signals[chest.get_signal_index('Accelerometer z')],
                                                       epoch_length=1,
                                                       sample_rate=
                                                       chest.signal_headers[chest.get_signal_index('Accelerometer x')][
                                                           'sample_rate'],
                                                       start_datetime=chest.header[start_key],
                                                       lowpass=12)[0]
                        subj_dict['chest_avm'] = crop_df(df=subj_dict['chest_avm'], start=row.start_time, end=row.end_time)

                        chest_avm.to_csv(row.chest_avm_file, index=False)

                        del chest
                        df_status.loc[df_status['full_id'] == row.full_id, 'chest_avm_file'] = 'success'

                    except:
                        df_status.loc[df_status['full_id'] == row.full_id, 'ankle_avm_file'] = 'failed'

                # create avm_combined
                try:
                    from Preprocessing import combine_df_avm
                    subj_dict['avm'] = combine_df_avm(full_id=row.full_id,
                                                      wrist_file=row.wrist_avm_file,
                                                      ankle_file=row.ankle_avm_file,
                                                      chest_file=row.chest_avm_file,
                                                      output_filename=row.avm_combined_file,
                                                      save_file=not os.path.exists(row.avm_combined_file))
                    subj_dict['avm'] = convert_time_columns(subj_dict['avm'])
                    subj_dict['avm'] = crop_df(df=subj_dict['avm'], start=row.start_time, end=row.end_time)


                except:
                    df_status.loc[df_status['full_id'] == row.full_id, 'avm_combined_file'] = 'failed'

            # Non-wear ---------------------------------------
            if row.wrist_nw:
                subj_dict['wrist_nw'] = pd.read_csv(row.wrist_nw_file)

                if 'event' in subj_dict['wrist_nw'].columns:
                    subj_dict['wrist_nw'] = subj_dict['wrist_nw'].loc[subj_dict['wrist_nw']['event'] == 'nonwear']
                    subj_dict['wrist_nw'].reset_index(drop=True, inplace=True)

                subj_dict['wrist_nw'] = convert_time_columns(subj_dict['wrist_nw'])
                subj_dict['wrist_nw'] = crop_df_bouts(df=subj_dict['wrist_nw'], start=row.start_time, end=row.end_time)

            if row.ankle_nw:
                subj_dict['ankle_nw'] = pd.read_csv(row.ankle_nw_file)

                if 'event' in subj_dict['ankle_nw'].columns:
                    subj_dict['ankle_nw'] = subj_dict['ankle_nw'].loc[subj_dict['ankle_nw']['event'] == 'nonwear']
                    subj_dict['ankle_nw'].reset_index(drop=True, inplace=True)

                subj_dict['ankle_nw'] = convert_time_columns(subj_dict['ankle_nw'])
                subj_dict['ankle_nw'] = crop_df_bouts(df=subj_dict['ankle_nw'], start=row.start_time, end=row.end_time)

            if row.chest_nw:
                subj_dict['chest_nw'] = pd.read_csv(row.chest_nw_file)

                if 'event' in subj_dict['chest_nw'].columns:
                    subj_dict['chest_nw'] = subj_dict['chest_nw'].loc[subj_dict['chest_nw']['event'] == 'nonwear']
                    subj_dict['chest_nw'].reset_index(drop=True, inplace=True)

                subj_dict['chest_nw'] = convert_time_columns(subj_dict['chest_nw'])
                subj_dict['chest_nw'] = crop_df_bouts(df=subj_dict['chest_nw'], start=row.start_time, end=row.end_time)

            if row.chest_weber_nw:
                subj_dict['chest_weber_nw'] = pd.read_csv(row.chest_weber_nw_file)
                subj_dict['chest_weber_nw'] = convert_time_columns(subj_dict['chest_weber_nw'])

                if not incl_ignored_weber_nw:
                    subj_dict['chest_weber_nw'] = subj_dict['chest_weber_nw'].loc[~subj_dict['chest_weber_nw']['ignore']]
                    subj_dict['chest_weber_nw'].reset_index(drop=True, inplace=True)

                subj_dict['chest_weber_nw'] = crop_df_bouts(df=subj_dict['chest_weber_nw'], start=row.start_time, end=row.end_time)

            # Epoching: 1-sec ---------------------------------------
            from ProcessData import create_df_context

            try:
                subj_dict['df1s'] = create_df_context(subj_dict=subj_dict,
                                                      min_gait_dur=5,
                                                      min_cadence=40,
                                                      pad_nonwear_bouts=nw_pad_mins)

                if 'posture_use' not in subj_dict['df1s'].columns:
                    subj_dict["df1s"] = flag_posture_transitions(df_epoch1s=subj_dict['df1s'],
                                                                 flag_n_secs=2)

                subj_dict['df1s'].to_csv(row.epoch1s_file, index=False)

            except:
                df_status.loc[df_status['full_id'] == row.full_id, 'epoch1s_file'] = 'failed'

            # Epoching: medium ---------------------------------------

            if not row.epoch_med:
                try:
                    print("Processing medium epoch length data")
                    subj_dict['epoch_med'] = reepoch_data(full_id=row.full_id,
                                                          df_epoch1s=subj_dict['df1s'],
                                                          new_epoch_len=epoch_med,
                                                          cutpoints=None,
                                                          snr_thresh=snr_thresh,
                                                          avm_thresh=avm_thresh,
                                                          quiet=True)

                    subj_dict['epoch_med'].to_csv(row.epoch_med_file, index=False)
                    df_status.loc[df_status['full_id'] == row.full_id, 'epoch_med_file'] = 'success'
                    row.epoch_med = True

                except:
                    df_status.loc[df_status['full_id'] == row.full_id, 'epoch_med_file'] = 'failed'

            if row.epoch_med:
                subj_dict['epoch_med'] = pd.read_csv(row.epoch_med_file)
                subj_dict['epoch_med'] = convert_time_columns(subj_dict['epoch_med'])
                subj_dict['epoch_med'] = crop_df_bouts(df=subj_dict['epoch_med'], start=row.start_time, end=row.end_time)

            # Epoching: long ---------------------------------------

            if not row.epoch_long:
                try:
                    print("Processing long epoch length data")
                    subj_dict['epoch_long'] = reepoch_data(full_id=row.full_id,
                                                           df_epoch1s=subj_dict['df1s'],
                                                           new_epoch_len=epoch_long,
                                                           cutpoints=None,
                                                           snr_thresh=snr_thresh,
                                                           avm_thresh=avm_thresh,
                                                           quiet=True)

                    subj_dict['epoch_long'].to_csv(row.epoch_long_file, index=False)

                    df_status.loc[df_status['full_id'] == row.full_id, 'epoch_long_file'] = 'success'
                    row.epoch_long = True

                except:
                    df_status.loc[df_status['full_id'] == row.full_id, 'epoch_long_file'] = 'failed'

            if row.epoch_long:
                subj_dict['epoch_long'] = pd.read_csv(row.epoch_long_file)
                subj_dict['epoch_long'] = convert_time_columns(subj_dict['epoch_long'])
                subj_dict['epoch_long'] = crop_df_bouts(df=subj_dict['epoch_long'], start=row.start_time, end=row.end_time)

        if return_data:
            subjs_dict[row.full_id] = subj_dict

    return subjs_dict, df_status
