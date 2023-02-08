
class AllData_old:

    def __init__(self, full_id="",
                 snr_edf_folder="",
                 edf_folder="",
                 df_summary=None,
                 med_epoch_len=5,
                 long_epoch_len=15*60,
                 min_gait_dur=0,
                 min_cadence=0,
                 snr_thresh=(5, 18),
                 sleepbouts_file=None,
                 sptw_file=None,
                 gaitbouts_file=None,
                 df_avm_file=None,
                 posture_file=None,
                 wrist_nw_file=None,
                 ankle_nw_file=None,
                 bittium_nw_file=None,
                 pad_nonwear_bouts=5,
                 avm_active_thresh=15,
                 flag_n_secs_transitions=2,
                 df_1s_filename=None,
                 start_time=None,
                 end_time=None):

        self.raw = {'ts': [], 'snr': [], 'ecg': [], 'acc_x': [], 'acc_y': [], 'acc_z': []}
        self.df_epoch = pd.DataFrame()
        self.df_epoch_med = pd.DataFrame()
        self.long_epoch_len = long_epoch_len
        self.med_epoch_len = med_epoch_len
        self.min_gait_dur = min_gait_dur
        self.min_cadence = min_cadence
        self.avm_active_thresh = avm_active_thresh

        self.start_time = start_time
        self.end_time = end_time

        self.snr_thresh = snr_thresh

        self.slope = None
        self.y_int = None

        self.full_id = full_id
        self.snr_edf_folder = snr_edf_folder
        self.edf_folder = edf_folder

        self.df_summary = df_summary

        self.df_epoch = pd.read_csv(df_avm_file) if os.path.exists(df_avm_file) else pd.DataFrame()

        if df_1s_filename is None:
            print("\nGenerating data from new...")

            if self.df_summary is None:
                self.df_summary, self.raw['snr'], self.raw['ts'] = import_raw(full_id=full_id,
                                                                              check_header=True,
                                                                              df_summary=df_summary)

            if self.df_summary is not None:
                self.df_summary, self.raw['snr'], self.raw['ts'] = import_raw(full_id=full_id,
                                                                              check_header=False,
                                                                              df_summary=df_summary)

            self.df_summary = self.df_summary.iloc[0]

            self.df_epoch, self.events = self.create_df_context(sleep_file=sleepbouts_file,
                                                                sptw_file=sptw_file,
                                                                gait_file=gaitbouts_file,
                                                                avm_file=df_avm_file,
                                                                ankle_nw_file=ankle_nw_file,
                                                                wrist_nw_file=wrist_nw_file,
                                                                posture_file=posture_file,
                                                                bittium_nw_file=bittium_nw_file,
                                                                df_epoch=self.df_epoch,
                                                                min_gait_dur=self.min_gait_dur,
                                                                min_cadence=self.min_cadence,
                                                                pad_nonwear_bouts=pad_nonwear_bouts)

            if self.start_time is not None:
                self.df_epoch = self.df_epoch.loc[self.df_epoch['start_time'] >= self.start_time]
            if self.end_time is not None:
                self.df_epoch = self.df_epoch.loc[self.df_epoch['start_time'] <= self.end_time]

            self.df_epoch['full_id'] = [full_id] * self.df_epoch.shape[0]

            self.df_epoch = flag_posture_transitions(self.df_epoch, flag_n_secs=flag_n_secs_transitions)

            self.df_posture_bouts = bout_posture(self.df_epoch)

            self.df_epoch_long = average_snr(snr_signal=self.raw['snr'], timestamps=self.raw['ts'], df_epoch=self.df_epoch,
                                             sample_rate=self.df_summary['sample_rate'], n_secs=self.long_epoch_len)

        if df_1s_filename is not None:
            print("\nReading data from csv...")
            self.df_epoch = pd.read_csv(df_1s_filename)
            self.df_epoch['start_time'] = pd.to_datetime(self.df_epoch['start_time'])

            if self.start_time is not None:
                self.df_epoch = self.df_epoch.loc[self.df_epoch['start_time'] >= self.start_time]
            if self.end_time is not None:
                self.df_epoch = self.df_epoch.loc[self.df_epoch['start_time'] <= self.end_time]

        # self.generate_df_epoch_med(epoch_len=med_epoch_len, avm_thresh=avm_active_thresh)
        self.df_epoch_med = reepoch_data(df_epoch1s=self.df_epoch, new_epoch_len=med_epoch_len,
                                         cutpoints=None, avm_thresh=avm_active_thresh)

        self.df_epoch_long = reepoch_data(df_epoch1s=self.df_epoch, new_epoch_len=long_epoch_len,
                                          cutpoints=None, avm_thresh=avm_active_thresh)

        self.reg = np.polyfit(self.df_epoch_long.loc[self.df_epoch_long['chest_nw_percent'] == 0]['days'],
                              self.df_epoch_long.loc[self.df_epoch_long['chest_nw_percent'] == 0]['snr'], deg=1)

    def plot_epoched(self):

        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(self.df_epoch['days'], self.df_epoch['snr'], color='black')
        ax.plot(self.df_epoch['days'], [i * self.reg[0] + self.reg[1] for i in self.df_epoch['days']],
                color='red', label=f"dB = {self.reg[0]:.3f}days + {self.reg[1]:.3f}")
        ax.legend()
        ax.set_ylabel("dB")
        ax.set_xlabel("Days into collection")
        plt.grid()

    @staticmethod
    def create_df_context(df_epoch, sleep_file, sptw_file, gait_file, avm_file,
                          ankle_nw_file, wrist_nw_file, bittium_nw_file, posture_file,
                          min_gait_dur=0, min_cadence=0, pad_nonwear_bouts=5):

        df_gait = pd.DataFrame()
        df_sleep = pd.DataFrame()
        sptw = pd.DataFrame()
        df_avm = pd.DataFrame()
        wrist_nw = pd.DataFrame()
        ankle_nw = pd.DataFrame()
        chest_nw = pd.DataFrame()
        df_posture = pd.DataFrame()

        epoch_len = int((df_epoch.iloc[1]['start_time'] - df_epoch.iloc[0]['start_time']).total_seconds())

        print("\nChecking tabular data files to provide context...")
        df_epoch = df_epoch.copy()

        if avm_file is not None:
            print("-Importing IMU AVM data...")
            df_avm = pd.read_csv(avm_file)
            df_avm['start_time'] = pd.to_datetime(df_avm['start_time'])

            # 1-sec epoch cropping
            start = max([df_epoch['start_time'].iloc[0], df_avm['start_time'].iloc[0]])
            end = min([df_epoch['start_time'].iloc[-1], df_avm['start_time'].iloc[-1]])
            df_epoch = df_epoch.loc[(df_epoch['start_time'] >= start) &
                                    (df_epoch['start_time'] <= end)].reset_index(drop=True)
            df_avm = df_avm.loc[(df_avm['start_time'] >= start) &
                                (df_avm['start_time'] <= end)].reset_index(drop=True)

        df_epoch['sleep_mask'] = [None] * df_epoch.shape[0]  # binary sleep bouts: sleep (1), wake (0)
        df_epoch['sptw_mask'] = [None] * df_epoch.shape[0]  # binary sleep period time window: sleep (1), wake (0)

        df_epoch['gait_mask'] = [None] * df_epoch.shape[0]  # binary gait: gait (1), no gait (0)

        df_epoch['posture'] = [None] * df_epoch.shape[0]  # postures as str
        df_epoch['upright_mask'] = [None] * df_epoch.shape[0]  # upright posture: upright (1), other (0)
        df_epoch['supine_mask'] = [None] * df_epoch.shape[0]  # supine posture: supine (1), other (0)
        df_epoch['prone_mask'] = [None] * df_epoch.shape[0]  # prone posture: prone (1), other (0)
        df_epoch['side_mask'] = [None] * df_epoch.shape[0]  # side lying posture: side lying (1), other (0)

        df_epoch['wrist_nw_mask'] = [None] * df_epoch.shape[0]
        df_epoch['ankle_nw_mask'] = [None] * df_epoch.shape[0]
        df_epoch['chest_nw_mask'] = [None] * df_epoch.shape[0]

        df_epoch['wrist_avm'] = [None] * df_epoch.shape[0]
        df_epoch['ankle_avm'] = [None] * df_epoch.shape[0]
        df_epoch['chest_avm'] = [None] * df_epoch.shape[0]

        if sleep_file is not None:
            print("-Importing sleep data...")
            df_sleep = pd.read_csv(sleep_file)
            df_sleep['start_time'] = pd.to_datetime(df_sleep['start_time'])
            df_sleep['end_time'] = pd.to_datetime(df_sleep['end_time'])
            df_epoch['sleep_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],
                                                 n_rows=df_epoch.shape[0], df=df_sleep, epoch_len=epoch_len)

        if sptw_file is not None:
            print("-Importing sleep period time window data...")
            sptw = pd.read_csv(sptw_file)
            sptw['start_time'] = pd.to_datetime(sptw['start_time'])
            sptw['end_time'] = pd.to_datetime(sptw['end_time'])
            df_epoch['sptw_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],
                                                n_rows=df_epoch.shape[0], df=sptw, epoch_len=epoch_len)

        if gait_file is not None:
            print("-Importing gait data...")

            df_gait = pd.read_csv(gait_file)
            df_gait['start_time'] = pd.to_datetime(df_gait['start_timestamp'] if
                                                   'start_timestamp' in df_gait.columns else df_gait['start_time'])
            df_gait['end_time'] = pd.to_datetime(df_gait['end_timestamp'] if
                                                 'end_timestamp' in df_gait.columns else df_gait['end_time'])
            df_gait = df_gait[[i for i in df_gait.columns if i not in ['start_timestamp', 'end_timestamp']]]
            df_gait['duration'] = [(row.end_time - row.start_time).total_seconds() for row in df_gait.itertuples()]
            df_gait['cadence'] = df_gait['step_count'] / df_gait['duration'] * 60

            if min_gait_dur > 0:
                print(f"    -Omitting bouts shorter than {min_gait_dur} seconds")
                df_gait = df_gait.loc[df_gait['duration'] >= min_gait_dur]
                df_gait.reset_index(drop=True, inplace=True)

            if min_cadence > 0:
                print(f"    -Omitting bouts with average cadence below {min_cadence} steps/min")
                df_gait = df_gait.loc[df_gait['cadence'] >= min_cadence]
                df_gait.reset_index(drop=True, inplace=True)

            df_epoch['gait_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],
                                                n_rows=df_epoch.shape[0], df=df_gait, epoch_len=epoch_len)

        if avm_file is not None:
            df_epoch['wrist_avm'] = list(df_avm['wrist_avm'])
            df_epoch['ankle_avm'] = list(df_avm['ankle_avm'])
            df_epoch['chest_avm'] = list(df_avm['chest_avm']) if 'chest_avm' in \
                                                                 df_avm.columns else [None] * df_epoch.shape[0]

        if posture_file is not None:
            print("-Importing posture data...")
            df_posture = import_posture(posture_file)
            df_posture['code'] = df_posture['posture'].replace({"upright": 1, 'supine': 2, 'prone': 3,
                                                                'rightside': 4, 'leftside': 5,
                                                                'reclined': 6, 'other': 7})
            code_dict = {1: 'upright', 2: 'supine', 3: 'prone', 4: 'rightside',
                         5: 'leftside', 6: 'reclined', 7: 'other', 0.0: 'other'}

            df_epoch['upright_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],  n_rows=df_epoch.shape[0],
                                                   df=df_posture.loc[df_posture['posture'] == 'upright'],
                                                   epoch_len=epoch_len)

            df_epoch['supine_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],  n_rows=df_epoch.shape[0],
                                                  df=df_posture.loc[df_posture['posture'] == 'supine'],
                                                  epoch_len=epoch_len)

            df_epoch['prone_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],  n_rows=df_epoch.shape[0],
                                                 df=df_posture.loc[df_posture['posture'] == 'prone'],
                                                 epoch_len=epoch_len)

            df_epoch['side_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],  n_rows=df_epoch.shape[0],
                                                df=df_posture.loc[df_posture['posture'].isin(['leftside', 'rightside'])],
                                                epoch_len=epoch_len)

            # postures as str. this is awful code. don't judge me.
            mask = np.zeros(df_epoch.shape[0])
            start_time = df_epoch['start_time'].iloc[0]

            for row in df_posture.itertuples():
                try:
                    start_i = int((row.start_time - start_time).total_seconds() / epoch_len)
                    end_i = int((row.end_time - start_time).total_seconds() / epoch_len)
                    mask[start_i:end_i] = row.code

                except AttributeError:
                    start_i = int((row.start_time - start_time).total_seconds() / epoch_len)
                    mask[start_i] = row.code

            df_epoch['posture'] = [code_dict[i] for i in mask]
            df_epoch['posture'].replace({"rightside": 'sidelying', 'leftside': 'sidelying', 'reclined': 'supine'},
                                             inplace=True)

        if bittium_nw_file is not None:
            print("-Importing Bittium non-wear data...")
            chest_nw = pd.read_csv(bittium_nw_file)
            chest_nw['start_time'] = pd.to_datetime(chest_nw['start_time'])
            chest_nw['end_time'] = pd.to_datetime(chest_nw['end_time'])

            if pad_nonwear_bouts > 0:
                print(f"-Padding nonwear bouts by {pad_nonwear_bouts} minutes")
                chest_nw['start_time'] -= timedelta(minutes=pad_nonwear_bouts)
                chest_nw['end_time'] += timedelta(minutes=pad_nonwear_bouts)

            df_epoch['chest_nw_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],
                                                    n_rows=df_epoch.shape[0], df=chest_nw, epoch_len=epoch_len)

        if ankle_nw_file is not None:
            print("-Importing ankle non-wear data...")
            ankle_nw = pd.read_csv(ankle_nw_file)
            ankle_nw['start_time'] = pd.to_datetime(ankle_nw['start_time'])
            ankle_nw['end_time'] = pd.to_datetime(ankle_nw['end_time'])
            df_epoch['ankle_nw_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],
                                                    n_rows=df_epoch.shape[0], df=ankle_nw, epoch_len=epoch_len)

        if wrist_nw_file is not None:
            print("-Importing wrist non-wear data...")
            wrist_nw = pd.read_csv(wrist_nw_file)
            wrist_nw['start_time'] = pd.to_datetime(wrist_nw['start_time'])
            wrist_nw['end_time'] = pd.to_datetime(wrist_nw['end_time'])
            df_epoch['wrist_nw_mask'] = create_mask(start_time=df_epoch['start_time'].iloc[0],
                                                    n_rows=df_epoch.shape[0], df=wrist_nw, epoch_len=epoch_len)

        print("Complete.")

        out_dict = {'gait': df_gait, 'sleep': df_sleep, 'sptw': sptw, 'posture': df_posture,
                    'avm': df_avm, 'wrist_nw': wrist_nw, 'ankle_nw': ankle_nw, 'chest_nw': chest_nw}

        return df_epoch, out_dict

    def plot_context(self, colnames=()):

        n_plots = len(colnames) + 1

        gridspec = [1]
        for i in range(len(colnames)):
            gridspec.append(.5)

        fig, ax = plt.subplots(n_plots, sharex='col', figsize=(12, 8), gridspec_kw={'height_ratios': gridspec})

        ax[0].plot(self.df_epoch['start_time'], self.df_epoch['snr'], color='black')
        ax[0].set_ylabel("dB")
        ax[0].grid()

        for i, col in enumerate(colnames):
            ax[i+1].plot(self.df_epoch['start_time'], self.df_epoch[col], color='black')
            ax[i+1].set_ylabel(col)
            ax[i+1].set_yticks([0, 1])

        plt.tight_layout()

        return fig


def average_snr_old(snr_signal, sample_rate, timestamps, df_epoch=None, n_secs=15):

    avg = [np.mean(snr_signal[i:i+int(sample_rate*n_secs)]) for
           i in np.arange(0, len(snr_signal), int(sample_rate*n_secs))]

    wrist_avm = [np.mean(df_epoch.loc[i:i + n_secs]['wrist_avm']) for i in np.arange(0, df_epoch.shape[0], n_secs)]
    ankle_avm = [np.mean(df_epoch.loc[i:i+n_secs]['ankle_avm']) for i in np.arange(0, df_epoch.shape[0], n_secs)]
    chest_avm = [np.mean(df_epoch.loc[i:i+n_secs]['chest_avm']) for i in np.arange(0, df_epoch.shape[0], n_secs)]

    timestamps = timestamps[::int(sample_rate * n_secs)]
    days = np.arange(0, len(snr_signal), int(sample_rate*n_secs)) / 86400 / sample_rate

    df_out = pd.DataFrame({"start_time": timestamps, 'days': days,
                           'snr': avg, 'wrist_avm': wrist_avm, 'ankle_avm': ankle_avm, 'chest_avm': chest_avm})

    if df_epoch is not None:
        chest_nwp = []
        ankle_nwp = []
        wrist_nwp = []
        for row in df_out.itertuples():
            df_use = df_epoch.loc[(df_epoch['start_time'] >= row.start_time) &
                                  (df_epoch['start_time'] < row.start_time + td(seconds=n_secs))]
            n_chest = list(df_use['chest_nw_mask']).count(1)
            n_ankle = list(df_use['ankle_nw_mask']).count(1)
            n_wrist = list(df_use['wrist_nw_mask']).count(1)

            n = df_use.shape[0]
            chest_nwp.append(round(100*n_chest/n, 1))
            ankle_nwp.append(round(100*n_ankle/n, 1))
            wrist_nwp.append(round(100*n_wrist/n, 1))

        df_out['chest_nw_percent'] = chest_nwp
        df_out['ankle_percent_nw'] = ankle_nwp
        df_out['wrist_percent_nw'] = wrist_nwp

    return df_out


def combine_logs(df_demos: pd.DataFrame(),
                 act_log_file: str = "W:/OND09 (HANDDS-ONT)/Digitized logs/handds_activity_log.xlsx",
                 nw_log_file: str = "W:/OND09 (HANDDS-ONT)/Digitized logs/handds_device_removal_log.xlsx"):

    # activity log -------------------------
    df_actlog = pd.read_excel(act_log_file)
    df_actlog = df_actlog.loc[df_actlog['subject_id'].isin(list(df_demos['subject_id']))]
    df_actlog['activity'].fillna("", inplace=True)

    # spaces are meant to be after 'bath' to exclude 'bathroom'
    df_actlog['water_activity'] = ['shower' in row.activity or 'Shower' in row.activity or 'bath ' in row.activity or
                                   'Bath ' in row.activity or 'Swim' in row.activity or 'swim' in row.activity
                                   for row in df_actlog.itertuples()]
    df_actlog_water = df_actlog.loc[df_actlog['water_activity']].reset_index(drop=True)

    df_actlog_water_formatted = df_actlog_water[['subject_id', 'activity', 'start_time', 'Notes.']]
    df_actlog_water_formatted['duration'] = df_actlog['duration (min)'].fillna("")
    df_actlog_water_formatted['end_time'] = [row.start_time + td(minutes=row.duration) if type(row.duration) is int else None
                                             for row in df_actlog_water_formatted.itertuples()]
    df_actlog_water_formatted = df_actlog_water_formatted[['subject_id', 'activity', 'start_time', 'end_time', 'Notes.']]
    df_actlog_water_formatted.columns = ['subject_id', 'activity', 'start_time', 'end_time', 'notes']

    # nonwear log ---------------------------
    df_nw_log = pd.read_excel(nw_log_file)
    df_nw_log = df_nw_log.loc[df_nw_log['SUBJECT'].isin(list(df_demos['subject_id']))]
    df_nw_log.columns = ['study_code', 'subject_id', 'coll_id', 'site', 'sensor_la', 'sensor_ra',
                         'sensor_lw', 'sensor_rw', 'sensor_bf', 'reason', 'time_removed',
                         'time_reattached', 'notes', 'Unnamed: 13']
    df_nw_log.fillna("", inplace=True)
    df_nw_log.replace({"Yes": 'yes', 'YES': 'yes'}, inplace=True)
    df_nw_log = df_nw_log.loc[df_nw_log['sensor_bf'] == 'yes']

    # spaces are meant to be after 'bath' to exclude 'bathroom'
    df_nw_log['water_activity'] = ['shower' in row.reason or 'Shower' in row.reason or 'bath ' in row.reason or
                                   'Bath ' in row.reason or 'Swim' in row.reason or 'swim' in row.reason
                                   for row in df_nw_log.itertuples()]
    df_nw_water = df_nw_log.loc[df_nw_log['water_activity']].reset_index(drop=True)

    df_nw_water_formatted = df_nw_water[['subject_id', 'reason', 'time_removed', 'time_reattached', 'notes']]
    df_nw_water_formatted.columns = ['subject_id', 'activity', 'start_time', 'end_time', 'notes']

    # combine ---------------
    df_log = pd.concat([df_actlog_water_formatted, df_nw_water_formatted]).sort_values(['subject_id', "start_time"])
    df_log['notes'].fillna("", inplace=True)

    df_log.reset_index(drop=True)

    return df_log, df_nw_log
