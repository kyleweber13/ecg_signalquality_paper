import os
import matplotlib.pyplot as plt
import pandas as pd
from ProcessData import AllData, create_collection_timestamps, get_collection_details, bin_avm_data
from Preprocessing import *
from Analysis import value_count_percent_grouping, generate_df_groupby_summary
import Plotting


def process_subject(full_id: str,
                    df_subjs: pd.DataFrame,
                    df_timestamps: pd.DataFrame,
                    load_ecg: bool = False,
                    avm_active_thresh: int or float = 15,
                    df_1s_filename: str or None = None,
                    df_snr_filename: str or None = None,
                    min_gait_dur: int or float = 5,
                    min_cadence: int or float = 0,
                    medium_epoch_len: int = 5,
                    long_epoch_len: int = 900):
    """ Function that formats file names and calls processing function class ProcessData.AllData.

        Parameters
        ----------
        full_id
            participant ID ({study_code}_{site}{participant ID} with no collection ID)
        load_ecg
            if Ture, loads ECG EDF file
        df_subjs
            dataframe containing participant IDs, start and stop times
        df_timestamps
            dataframe containing collection periods accounting for electrode reapplication
        avm_active_thresh
            average vector magnitude (AVM) threshold for classifying activity from inactivity
        df_1s_filename
            full pathway to already processed 1-second epoch csv file
        df_snr_filename
            pathway to 1-second SNR average csv file
        min_gait_dur
            minimum gait bout duration in flagging periods as gait or not
        min_cadence
            minimum gait bout cadence in flagging periods as gait or not
        medium_epoch_len
            medium epoch length in seconds. Used for primary analysis
        long_epoch_len
            long epoch length in seconds. Used for quality over time trend analysis

        Returns
        -------
        ProcessData.AllData class instance
    """

    # rows from dfs for current participant
    df_subj = df_subjs.loc[df_subjs['full_id'] == full_id]
    df_timestamp = df_timestamps.loc[df_timestamps['full_id'] == full_id]

    # nonwear filenames (if they exist) ----------

    ankle_nw_file = f"W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{full_id}_01_AXV6_LAnkle_NONWEAR.csv"
    ankle_nw_file = ankle_nw_file if os.path.exists(ankle_nw_file) else \
        f"W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{full_id}_01_AXV6_RAnkle_NONWEAR.csv"

    wrist_nw_file = f"W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{full_id}_01_AXV6_LWrist_NONWEAR.csv"
    wrist_nw_file = wrist_nw_file if os.path.exists(wrist_nw_file) else \
        f"W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{full_id}_01_AXV6_RWrist_NONWEAR.csv"

    # posture filename (if it exists) -----------
    posture_file = f"{paper_dir}Data/chest_posture/{full_id}_posture.csv" if \
                       os.path.exists(f"{paper_dir}Data/chest_posture/{full_id}_posture.csv") else None

    data = AllData(full_id=full_id,
                   df_1s_filename=df_1s_filename,
                   snr_edf_folder="W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/",
                   edf_folder='W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/',
                   load_ecg=load_ecg,
                   df_subj=df_subj,
                   sleepbouts_file=f"W:/NiMBaLWEAR/OND09/analytics/sleep/bouts/{full_id}_01_SLEEP_BOUTS.csv",
                   sptw_file=f"W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/{full_id}_01_SPTW.csv",
                   gaitbouts_file=f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/{full_id}_01_GAIT_BOUTS.csv",
                   min_gait_dur=min_gait_dur,
                   min_cadence=min_cadence,
                   # df_avm_file=f"{paper_dir}Data/dev_data/avm/{full_id}_avm.csv" if os.path.exists(f"{paper_dir}Data/dev_data/avm/{full_id}_avm.csv") else "",
                   df_snr_filename=df_snr_filename,
                   posture_file=posture_file,
                   ankle_nw_file=ankle_nw_file,
                   wrist_nw_file=wrist_nw_file,
                   bittium_nw_file=f"{paper_dir}Data/dev_data/bittium_nonwear/{full_id}_bittium_nonwear_test.csv" if \
                                   df_subj['proc_nw'].iloc[0] else "",
                   pad_nonwear_bouts=5,
                   flag_n_secs_transitions=2,
                   avm_active_thresh=avm_active_thresh,
                   med_epoch_len=medium_epoch_len,
                   long_epoch_len=long_epoch_len,
                   start_time=df_timestamp.iloc[0]['use_start'],
                   end_time=df_timestamp.iloc[0]['use_end'],
                   )

    return data


""" ============================================== STUDY SET-UP =================================================== """
paper_dir = 'O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/'

df_demos = pd.read_excel(f"{paper_dir}Data/OND09_Demos_Use.xlsx")
df_demos_use = df_demos.loc[(df_demos['bittium_orientation'] == 'vertical') & (df_demos['gait_aid'] == 'NONE') & (df_demos['moca'] >= 26) & (df_demos['cohort'] == 'NONE')]

df_removal = pd.read_excel(f"{paper_dir}Data/electrode_removals.xlsx")

"""
df_subjs = get_collection_details(df_subjs=df_demos_use, 
                                  nw_folder="C:/Users/ksweber/Desktop/ECG_nonwear_dev/FinalBouts_NoSNR/",
                                  edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
                                  snr_folder="W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/")
df_subjs.to_excel(f"{paper_dir}Data/CollectionSummaries.xlsx", index=False)
"""

df_subjs = pd.read_excel(f"{paper_dir}Data/CollectionSummaries.xlsx")

df_timestamps = create_collection_timestamps(df_subjs=df_subjs,
                                             df_timestamps=pd.read_excel(f"{paper_dir}Data/collection_timestamps.xlsx"))
# df_timestamps.to_excel(f"{paper_dir}Data/collection_timestamps.xlsx", index=False)

""" ========================================= INDIVIDUAL PARTICIPANT ============================================== """

full_id = 'OND09_0001'

data = process_subject(full_id=full_id,
                       load_ecg=False,
                       df_subjs=df_subjs,
                       df_timestamps=df_timestamps,
                       medium_epoch_len=5,
                       long_epoch_len=900,
                       avm_active_thresh=15,
                       df_1s_filename=f"{paper_dir}Data/dev_data/processed_epoch1s/{full_id}_epoch1s.csv",
                       # df_1s_filename=None,
                       df_snr_filename=f"{paper_dir}Data/dev_data/snr_1s/{full_id}_snr1s.csv"
                       )

# saving epoch dataframes --------
"""
data.df_epoch.to_csv(f"{paper_dir}Data/dev_data/processed_epoch1s/{full_id}_epoch1s.csv", index=False)
data.df_epoch_med.to_csv(f"{paper_dir}Data/dev_data/processed_epoch_med/{full_id}_epoch{data.med_epoch_len}.csv", index=False)
data.df_epoch_long.to_csv(f"{paper_dir}Data/dev_data/processed_epoch_long/{full_id}_epoch{data.long_epoch_len}.csv", index=False)
"""

""" ====================================== ANALYSIS ============================================================== """

# bins AVM data by specified bin size --------
# df_desc, df_values = bin_avm_data(df_epoch=data.df_epoch_med, binsize=10, ignore_nw=True, ignore_sleep=False, x='chest', y='ankle')

# Signal quality categories by grouping variable --------
# df_nepochs, df_percent = value_count_percent_grouping(df=data.df_epoch_med[data.df_epoch_med['all_wear']], colname='snr_hr', groupby='posture_use')

# ================================ PLOTTING ================================

# -------------------------- single participants ---------------------------

# Timeseries data ----------------------------------------------------------

# timeseries masks (binary) and SNR values (df_epoch and df_epoch_long)
# fig = data.plot_context(['gait_mask', 'sleep_mask', 'chest_nw_mask'])

# able to plot SNR, wrist/ankle/chest AVM and masks overlaid on appropriate subplots --------
# Plotting.plot_df_epoch(df_epoch=data.df_epoch, df_epoch_long=data.df_epoch_long, time_column='days', columns=['snr', 'sptw_mask', 'gait_mask', 'wrist_nw_mask', 'ankle_nw_mask', 'chest_nw_mask', 'wrist_avm', 'ankle_avm', 'chest_avm'])

# Descriptive data ---------------------------------------------------------

# colour-coded histogram of number of epochs by SNR by SNR category
# Plotting.generate_smital_region_histogram(df=data.df_epoch_med, thresh=data.snr_thresh, ignore_nw=True, shade_regions=True)

# Pie plot of SNR categories (%)
# fig = Plotting.pieplot_snr_categories(df=data.df_epoch_med, ignore_nw=True)

# Boxplots for dv_colname grouped by each level in columns --------
# fig = Plotting.generate_boxplots(columns=['posture_use'], dv_colname='snr', df=data.df_epoch_med)

# SNR by posture
# fig = Plotting.plot_density_by_posture(df_epoch=data.df_epoch_med, ignore_nw=True, colname=['posture_use'], thresholds=data.snr_thresh, xlim=(-5, 30))
# fig = Plotting.plot_snr_by_posture_barplot(df_epoch=data.df_epoch_med, ignore_nw=True, use_sem=False)

# chest/wrist/ankle AVM ~ SNR
# fig = Plotting.plot_avm_snr_scatter(df=data.df_epoch_long, min_avm=5, ignore_nw=True)

# summarizing grouped data
# df_g = generate_df_groupby_summary(df=data.df_epoch_med, groupby_column='posture_use', dv_column='snr_quality', missing_value=0, method='count')
# df_g = generate_df_groupby_summary(df=data.df_epoch_med, groupby_column='posture_use', dv_column='snr', missing_value=0, method='mean')

# -------------------------- multiple participants ---------------------------

# boxplot
# generate_boxplots(df=data.df_epoch15, dv_colname='snr', columns=['sleep_any', 'gait_any'], showfliers=False, scale_plotwidth=False)

# average SNR by time period across participants + error range + regression line
# plot_trend(df=df_all_1h, ci_type='sem', include_trend=True, slope=m, yint=b)

# Density plot of SNR grouped by specified column
# fig_density = plotdensity_snr_by_type(df_epoch=data.df_epoch15.replace({True: "sleep", False: 'awake'}), thresholds=(5, 18), ignore_nw=True, group_col='sleep_any', color_dict={'sleep': "navy", 'awake': 'orange'})
# fig_density = plotdensity_snr_by_type(df_epoch=df_all_15s.replace({True: "gait", False: 'no gait'}), thresholds=(5, 18), ignore_nw=True, group_col='gait_any', color_dict={'gait': "black", 'no gait': 'red'})
# fig_density = plotdensity_snr_by_type(df_epoch=df_all_15s, thresholds=(5, 18), ignore_nw=True, group_col='day_int', color_dict={1: 'red', 2: 'orange', 3: 'gold', 4: 'limegreen', 5: 'green', 6: 'dodgerblue', 7: 'blue', 8: 'purple', 9: 'grey'})

# heatmap for each participant's timeseries SNR + average across participants
# ptable, heatmap = trend_heatmap(df=df_all_15s, ignore_nw=False, use_full_id=False, include_avg=True)

# % of collection in each Smital category for each participant + average across participants
# subj_cats, cats_fig = plot_snr_cat_by_participant(df=df_all_15s, ignore_nw=True, use_full_id=False, incl_average=True, bar_width=.9)

# heatmap for wrist vs. ankle AVM SNR
# avm_heatmap(df_desc, norm_range=(5, 22))

# generate_boxplots(df=df_all_15s.loc[~np.isnan(df_all_15s['bitt_nw_percent'])], dv_colname='snr', columns=['day_int'], showfliers=False)

# SNR by AVM bin by device w/ regression and confidence interval
# df_avm, df_avm_stats, avm_fig = scatter_snr_by_avm(df=df_all_15s.loc[df_all_15s['chest_avm_bin'] >= 10], min_datapoints=25, thresholds=(5, 18), use_sem=False)


from ECG_SignalQuality_Paper.DataImport import import_snr_raw
from ECG_SignalQuality_Paper.ProcessData import average_snr

failed = []

for i in tqdm(range(df_subjs.shape[0])):
    row = df_subjs.loc[i]
    fname = f"O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch1s/{row.full_id}_epoch1s.csv"

    if not os.path.exists(fname):

        try:
            data = process_subject(full_id=row.full_id,
                                   load_ecg=False,
                                   df_subjs=df_subjs,
                                   df_timestamps=df_timestamps,
                                   medium_epoch_len=5,
                                   long_epoch_len=900,
                                   avm_active_thresh=15,
                                   # df_1s_filename=f"{paper_dir}Data/epoch1s/{row.full_id}_epoch1s.csv",
                                   df_1s_filename=None,
                                   df_snr_filename=f"{paper_dir}Data/dev_data/snr_1s/{row.full_id}_snr1s.csv"
                                   )

            data.df_epoch.to_csv(fname, index=False)
            data.df_epoch_med.to_csv(f"{paper_dir}Data/dev_data/processed_epoch_med/{full_id}_epoch{data.med_epoch_len}.csv", index=False)
            data.df_epoch_long.to_csv(f"{paper_dir}Data/dev_data/processed_epoch_long/{full_id}_epoch{data.long_epoch_len}.csv", index=False)
        except:
            failed.append(row.full_id)
