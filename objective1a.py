import matplotlib.pyplot as plt
from DataImport import combine_files
from Analysis import value_count_percent_grouping
from ProcessData import assign_snr_category
from Plotting import pieplot_snr_categories, generate_smital_region_histogram


def objective1a(df_files,
                epoch1s_folder="O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/processed_epoch1s/"):

    # imports data
    df1s_all = combine_files(folder=epoch1s_folder, full_ids=df_files['full_id'])
    df1s_all['full_id'] = [row.filename.split("_epoch1s.csv")[0] for row in df1s_all.itertuples()]
    df1s_all = assign_snr_category(df=df1s_all, thresholds=(5, 18))

    # removes any non-wear
    df1s_wear = df1s_all.loc[df1s_all['all_wear']]

    # SNR quality category tall and percent by participant
    tally, percent, percent_desc = value_count_percent_grouping(df=df1s_wear, colname="snr_quality", groupby='full_id')

    # pie plot of all SNR category data
    pieplot = pieplot_snr_categories(df=df1s_wear, ignore_nw=True)
    pieplot.axes[0].set_title(f"{len(df1s_wear['filename'].unique())} participants\n({df1s_wear.shape[0]} one-second epochs, chest non-wear removed..?)")

    # histogram of all quality categories
    hist = generate_smital_region_histogram(df=df1s_wear, thresh=(5, 18), ignore_nw=True, shade_regions=False, xrange=(-10, 30))
    hist.axes[0].set_title(f"{len(df1s_wear['filename'].unique())} participants\n({df1s_wear.shape[0]} one-second epochs, chest non-wear removed..?)")
    plt.tight_layout()

    return df1s_wear, tally, percent, pieplot, hist


# df1s_wear, tally, percent, pieplot, hist = objective1a(df_files.iloc[:10])