import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DataImport import combine_files
from Filtering import filter_signal
from ProcessData import flag_day_of_collection
from Analysis import describe_by_subj
import pingouin as pg
import seaborn as sb


def analyze_days_difference_usable_snr_bysubj(df, use_days=(0, 6), min_daily_count=40):

    def true_count(data):

        try:
            v = list(data).count(True) / len(data)
        except ZeroDivisionError:
            v = 0

        return v

    # percent of Q1/Q2 data by day by participant -----------------------------------------------
    df_hr = pd.DataFrame()

    for day in list(use_days):
        col = [true_count(df.loc[(df['full_id'] == full_id) & (df['day_int'] == day)]['snr_hr']) for
               full_id in df['full_id'].unique()]
        df_hr[f'day{day}'] = col

        df_hr[f"day{day}_count"] = [df.loc[(df['full_id'] == full_id) & (df['day_int'] == day)].shape[0] for
                                    full_id in df['full_id'].unique()]

    df_hr.insert(loc=0, value=df['full_id'].unique(), column='full_id')

    df_hr = df_hr.loc[(df_hr[f"day{use_days[0]}_count"] >= min_daily_count) &
                      (df_hr[f"day{use_days[1]}_count"] >= min_daily_count)]
    df_hr['diff'] = df_hr[f"day{use_days[1]}"] - df_hr[f"day{use_days[0]}"]

    t_hr = pg.ttest(paired=True, x=df_hr[f'day{use_days[0]}'], y=df_hr[f'day{use_days[1]}'])

    # percent of Q1 data by day by participant --------------------------------------------------
    df_q1 = pd.DataFrame()

    for day in list(use_days):
        col = [true_count(df.loc[(df['full_id'] == full_id) & (df['day_int'] == day)]['snr_quality'] == 'Q1') for
               full_id in df['full_id'].unique()]
        df_q1[f"day{day}"] = col

        df_q1[f"day{day}_count"] = [df.loc[(df['full_id'] == full_id) & (df['day_int'] == day)].shape[0] for
                                    full_id in df['full_id'].unique()]

    df_q1.insert(loc=0, value=df['full_id'].unique(), column='full_id')

    df_q1 = df_q1.loc[(df_q1[f"day{use_days[0]}_count"] >= min_daily_count) &
                      (df_q1[f"day{use_days[1]}_count"] >= min_daily_count)]
    df_q1['diff'] = df_q1[f"day{use_days[1]}"] - df_q1[f"day{use_days[0]}"]

    t_q1 = pg.ttest(paired=True, x=df_q1[f'day{use_days[0]}'], y=df_q1[f"day{use_days[1]}"])

    df_t = pd.concat([t_hr, t_q1])
    df_t.index = ['hr', 'q1']

    return df_hr, df_q1, df_t


df_long_all = combine_files(folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch_long/", full_ids=df_files['full_id'])

df_long_all = flag_day_of_collection(df_long_all)

df_long_all = df_long_all.loc[df_long_all['chest_nw_percent'] == 0]
# df_long_all = df_long_all.loc[df_long_all['sleep_percent'] == 0]

df_long_desc = describe_by_subj(df=df_long_all, stat='50%', dv_colname='snr', groupby='coll_day', subj_colname='full_id')

# df_long_all_grouped = df_long_all.groupby(["coll_day"])['snr'].describe()
# df_long_all_grouped['sem'] = df_long_all_grouped['std'] / np.sqrt(df_long_all_grouped['count'])
# df_long_all_grouped['ci95'] = df_long_all_grouped['sem'] * 1.96

# =============================
# df_all = combine_files(folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch_med/", full_ids=df_files['full_id'].iloc[:10])

df_hr, df_q1, df_t = analyze_days_difference_usable_snr_bysubj(df=df_long_all, use_days=[0, 6], min_daily_count=40)

# day x vs. day y SNR by participant
sb.catplot(data=df_long_all, x="day_int", y="snr", kind='violin')

df_hr_long = pd.melt(frame=df_hr, id_vars='full_id',
                     value_vars=['day0', 'day6'], value_name='hr_usable',
                     var_name='day')
