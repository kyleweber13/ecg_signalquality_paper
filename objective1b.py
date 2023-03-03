import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DataImport import combine_files
from ProcessData import flag_day_of_collection, participant_data_by_day
from Analysis import describe_by_subj
import pingouin as pg
import seaborn as sb
import scipy.stats
from tqdm import tqdm


def analyze_days_difference_usable_snr_bysubj(df, use_days=(0, 6), min_daily_count=40, parametric=None):

    def true_count(data):

        try:
            v = list(data).count(True) * 100 / len(data)
        except ZeroDivisionError:
            v = 0

        return v

    # percent of Q1/Q2 data by day by participant -----------------------------------------------
    df_hr = pd.DataFrame()

    for day in tqdm(list(use_days)):
        col = [true_count(df.loc[(df['full_id'] == full_id) & (df['day_int'] == day)]['snr_hr']) for
               full_id in df['full_id'].unique()]
        df_hr[f'day{day}'] = col

        df_hr[f"day{day}_count"] = [df.loc[(df['full_id'] == full_id) & (df['day_int'] == day)].shape[0] for
                                    full_id in df['full_id'].unique()]

    df_hr.insert(loc=0, value=df['full_id'].unique(), column='full_id')

    df_hr = df_hr.loc[(df_hr[f"day{use_days[0]}_count"] >= min_daily_count) &
                      (df_hr[f"day{use_days[1]}_count"] >= min_daily_count)]
    df_hr['diff'] = df_hr[f"day{use_days[1]}"] - df_hr[f"day{use_days[0]}"]

    # percent of Q1 data by day by participant --------------------------------------------------
    df_q1 = pd.DataFrame()

    for day in tqdm(list(use_days)):
        col = [true_count(df.loc[(df['full_id'] == full_id) & (df['day_int'] == day)]['snr_quality'] == 'Q1') for
               full_id in df['full_id'].unique()]
        df_q1[f"day{day}"] = col

        df_q1[f"day{day}_count"] = [df.loc[(df['full_id'] == full_id) & (df['day_int'] == day)].shape[0] for
                                    full_id in df['full_id'].unique()]

    df_q1.insert(loc=0, value=df['full_id'].unique(), column='full_id')

    df_q1 = df_q1.loc[(df_q1[f"day{use_days[0]}_count"] >= min_daily_count) &
                      (df_q1[f"day{use_days[1]}_count"] >= min_daily_count)]
    df_q1['diff'] = df_q1[f"day{use_days[1]}"] - df_q1[f"day{use_days[0]}"]

    # Statistical test ---------------------

    if parametric is None:
        hr0 = scipy.stats.shapiro(df_hr[f'day{use_days[0]}'])[1] >= .05
        hr1 = scipy.stats.shapiro(df_hr[f'day{use_days[1]}'])[1] >= .05
        q0 = scipy.stats.shapiro(df_q1[f'day{use_days[0]}'])[1] >= .05
        q1 = scipy.stats.shapiro(df_q1[f'day{use_days[1]}'])[1] >= .05

        parametric = hr0 and hr1 and q0 and q1

        print("Testing HR and Q1 distributions for normality to determine use of parametric vs. non-parametric...")
        print(f"-HR day #{use_days[0]}: {'non-' if not hr0 else ''}parametric")
        print(f"-HR day #{use_days[1]}: {'non-' if not hr0 else ''}parametric")
        print(f"-Q1 day #{use_days[0]}: {'non-' if not hr0 else ''}parametric")
        print(f"-Q1 day #{use_days[1]}: {'non-' if not hr0 else ''}parametric")
        print(f"Using {'non-' if not parametric else ''}parametric tests.")

    if parametric:
        t_hr = pg.ttest(paired=True, x=df_hr[f'day{use_days[0]}'], y=df_hr[f'day{use_days[1]}'])
        t_q1 = pg.ttest(paired=True, x=df_q1[f'day{use_days[0]}'], y=df_q1[f"day{use_days[1]}"])

    if not parametric:
        t_hr = pg.wilcoxon(x=df_hr[f'day{use_days[0]}'], y=df_hr[f'day{use_days[1]}'])
        t_q1 = pg.wilcoxon(x=df_q1[f'day{use_days[0]}'], y=df_q1[f"day{use_days[1]}"])

    df_t = pd.concat([t_hr, t_q1])
    df_t.index = ['hr', 'q1']
    df_t.insert(loc=0, column='test',
                value=['wilcoxon'] * df_t.shape[0] if not parametric else ['ttest'] * df_t.shape[0])

    return df_hr, df_q1, df_t


df_long_all = combine_files(folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch_med/",
                            full_ids=df_files.loc[df_files['effective_dur_h'] >= 168]['full_id'])

df_long_all = flag_day_of_collection(df_long_all)

df_long_all = df_long_all.loc[df_long_all['all_weber_wear']]
df_long_all.reset_index(drop=True, inplace=True)
# df_long_all = df_long_all.loc[df_long_all['sleep_percent'] == 0]

# df_long_desc = describe_by_subj(df=df_long_all, stat='50%', dv_colname='snr', groupby=['day_int'], subj_colname='full_id')
# df_long_desc_long = df_long_desc.melt(id_vars='full_id', value_name='snr').sort_values(["full_id", 'day_int']).reset_index(drop=True)
# df_long_all_grouped = df_long_all.groupby(["coll_day"])['snr'].describe()
# df_long_all_grouped['sem'] = df_long_all_grouped['std'] / np.sqrt(df_long_all_grouped['count'])
# df_long_all_grouped['ci95'] = df_long_all_grouped['sem'] * 1.96
# df_daily_snr = participant_data_by_day(df_combined=df_long_all, dv_colname='snr', day_colname='day_int', stat='50%')

# =============================
# df_all = combine_files(folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch_med/", full_ids=df_files['full_id'])

# df_hr, df_q1, df_stat = analyze_days_difference_usable_snr_bysubj(df=df_long_all, use_days=[1, 7], min_daily_count=40, parametric=None)
df_hr, df_q1, df_stat = analyze_days_difference_usable_snr_bysubj(df=df_long_all, use_days=[1, 7], min_daily_count=7200, parametric=None)

# day x vs. day y SNR by participant
# sb.catplot(data=df_long_all.loc[df_long_all['day_int'] < 7], x="day_int", y="snr", kind='violin')

fig, ax = plt.subplots(1, figsize=(8, 6))
ax.scatter(df_q1['day1'], df_q1['day7'], color='black', zorder=2)
ax.set_xlabel("Day #1 (% Q1 data)")
ax.set_ylabel("Day #7 (% Q1 data)")
ax.set_xlim(-1, 101)
ax.set_ylim(-1, 101)
ax.plot(np.arange(101), np.arange(101), color='black', linestyle='dashed', label='line of\nidentity', zorder=1)
ax.legend()
ax.grid(zorder=0)

ba_fig = pg.plot_blandaltman(df_q1['day1'], df_q1['day7'], xaxis='x', scatter_kws={'color': 'black'}, figsize=(8, 6), annotate=False)
plt.tight_layout()