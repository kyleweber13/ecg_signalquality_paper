import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
import seaborn as sns
import scipy.stats


def generate_smital_region_histogram(df: pd.DataFrame,
                                     thresh: list or tuple = (5, 18),
                                     ignore_nw: bool = True,
                                     shade_regions: bool = False,
                                     xrange: list or tuple or None = None):
    """ Generates colour-coded histogram of Smital SNR quality categories using data in df.

        Parameter
        ---------
        df
            dataframe to use. requires 'snr' column. also requires 'chest_nw_mask' column if ignore_nw == True
        thresh
            SNR thresholds to classify as Q1, Q2, Q3
        ignore_nw
            if True, ignores rows of data where 'chest_nw_mask' != 0
        shade_regions
            if True, shades vertical bars on graph corresponding to quality categories given by thresh
        xrange
            if not None, uses boundaries as histogram bins and x-axis limits

        Returns
        -------
        figure
    """

    df = df.copy()

    if ignore_nw:
        if 'chest_nw_mask' in df.columns:
            df = df.loc[df['chest_nw_mask'] == 0]
        if 'chest_nw_mask' not in df.columns:
            df = df.loc[df['chest_nw_percent'] == 0]

    thresh = sorted(thresh)

    if xrange is not None:
        xrange = sorted(xrange)
        bins = np.arange(xrange[0], xrange[1] + 2, 1)

        df.loc[df['snr'] < xrange[0], 'snr'] = xrange[0]
        df.loc[df['snr'] > xrange[1], 'snr'] = xrange[1] + 1

    lq = df.loc[df['snr'] < thresh[0]]
    mq = df.loc[(df['snr'] >= thresh[0]) & (df['snr'] < thresh[1])]
    hq = df.loc[df['snr'] >= thresh[1]]

    data_range = [int(np.floor(df['snr'].min())), int(np.ceil(df['snr'].max()))]

    if xrange is None:
        bins = np.arange(data_range[0], data_range[1], 1)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    lq['snr'].plot.hist(bins=bins, weights=100*np.ones(lq.shape[0])/df.shape[0], ax=ax,
                        color='red', edgecolor='black', label='Low quality', alpha=1)
    mq['snr'].plot.hist(bins=bins, weights=100*np.ones(mq.shape[0])/df.shape[0], ax=ax,
                        color='dodgerblue', edgecolor='black', label='Medium quality', alpha=1)
    hq['snr'].plot.hist(bins=bins, weights=100*np.ones(hq.shape[0])/df.shape[0], ax=ax,
                        color='limegreen', edgecolor='black', label='High quality', alpha=1)

    if shade_regions:
        ax.axvspan(xmin=ax.get_xlim()[0], xmax=thresh[0], ymin=0, ymax=1, color='red', alpha=.1)
        ax.axvspan(xmin=thresh[0], xmax=thresh[1], ymin=0, ymax=1, color='dodgerblue', alpha=.1)
        ax.axvspan(xmin=thresh[1], xmax=ax.get_xlim()[1], ymin=0, ymax=1, color='limegreen', alpha=.1)

    if xrange is None:
        ax.set_xlim(data_range)
    if xrange is not None:
        ax.set_xlim(xrange[0], xrange[1]+2)
        xticks = list(ax.get_xticks())[:-2]
        tick_int = xticks[1] - xticks[0]

        xticks = [str(int(i)) for i in xticks]
        xticks.append(f">{str(int(int(xticks[-1]) + tick_int))}")
        ax.set_xticklabels(xticks)

    ax.legend()
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Percent of epochs")
    plt.tight_layout()

    return fig


def generate_boxplots(df: pd.DataFrame,
                      dv_colname: str = 'snr',
                      columns: list or tuple = (),
                      showfliers: bool = False,
                      scale_plotwidth: bool = True):
    """ Generates boxplot of df_colname for each grouping column specified in 'columns'

        Parameters
        ----------
        df
            dataframe. requires columns specified by dv_colname and columns
        dv_colname
            outcome variable of interest
        columns
            grouping variable(s)
        showfliers
            boolean to show outliers
        scale_plotwidth
            if True, each subplot is scaled to the number of groups
            e.g., if plotting variables with 4 and 2 groups, the former subplot will be
            2/3rds of the plot's total width

        Returns
        -------
        figure

    """

    valid_cols = [i for i in columns if len(df[i].unique()) > 1]
    n_cols = len(valid_cols)

    col_groups = [len(df[i].unique()) for i in valid_cols] if scale_plotwidth else [1] * n_cols

    if len(valid_cols) < len(columns):
        print(f"Cannot create boxplot for {[i for i in columns if i not in valid_cols]}")

    fig, ax = plt.subplots(1, n_cols, sharey='row', figsize=(12, 8), gridspec_kw={'width_ratios': col_groups})

    if len(valid_cols) > 1:
        ax[0].set_ylabel(dv_colname)
    if len(valid_cols) == 1:
        ax.set_ylabel(dv_colname)

    for col, colname in enumerate(valid_cols):
        df.boxplot(by=colname, column=dv_colname, ax=ax[col] if len(valid_cols) > 1 else ax,
                   showfliers=showfliers, widths=.75,
                   boxprops={"color": 'black', 'linewidth': 1.5},
                   whiskerprops={"color": 'black', 'linewidth': 1.5},
                   medianprops={"color": 'red', 'linewidth': 1.5},
                   capprops={'color': 'black', "linewidth": 1.5})
        if len(valid_cols) > 1:
            ax[col].set_title(colname)
            ax[col].set_xlabel("")
        if len(valid_cols) == 1:
            ax.set_title(colname)
            ax.set_xlabel("")

        try:
            # binary variables
            if len(valid_cols) > 1:
                ax[col].set_xticklabels([f"not {colname}\n(N={df[colname].value_counts().loc[0]})",
                                         f"{colname}\n(N={df[colname].value_counts().loc[1]})"])
            if len(valid_cols) == 1:
                ax.set_xticklabels([f"not {colname}\n(N={df[colname].value_counts().loc[0]})",
                                    f"{colname}\n(N={df[colname].value_counts().loc[1]})"])
        except (KeyError, ValueError):
            pass

    n_participants = len(df['full_id'].unique())
    plt.suptitle(f"{dv_colname} grouped by binary states of {valid_cols} ({n_participants} "
                 f"participant{'s' if n_participants != 1 else ''})")
    plt.tight_layout()

    return fig


def plot_trend(df, ci_type='sem', include_trend=True, slope=0, yint=0):

    d = df.groupby('days')['snr'].describe()

    fig, ax = plt.subplots(1)
    ax.plot(d.index, d['mean'], color='black', label='mean')

    if ci_type in ['sem', 'SEM', "SE", 'se']:
        e = d['std']/np.sqrt(d['count'])

    ax.fill_between(d.index, d['mean'] + e, d['mean'] - e, color='grey', alpha=.5, label=ci_type)

    if include_trend:
        r = np.polyfit(df['days'], df['snr'], deg=1)
        ax.plot(d.index, [i * slope + yint for i in d.index],
                color='red', linestyle='dashed', label=f'y={slope:.3f}x + {yint:.3f}')

    ax.legend()
    ax.set_xlabel("Days")
    ax.set_ylabel("SNR")
    ax.set_xlim(-.1, d.index.max() + .1)

    plt.tight_layout()


def plot_df_epoch(df_epoch: pd.DataFrame,
                  df_epoch_long: pd.DataFrame or None = None,
                  time_column: str = 'start_time',
                  columns: list or tuple = ()):
    """ Plots all given data on appropriate axes given other data that's already been plotted

        Parameters
        ----------
        df_epoch
            dataframe containing all data to be plotted. requires columns specified in colums
        df_epoch_long
            option: df_epoch re-epoched into longer epochs. Only plots SNR
        time_column
            column name in df_epoch to use as x-axis. 'days' or 'start_time'
        columns
            list/tuple of columns in df to plot

        Returns
        -------
        figure
    """

    nonmask_cols = [i for i in columns if 'mask' not in i]
    mask_cols = [i for i in columns if 'mask' in i]

    ax_labels = {}

    fig, ax = plt.subplots(len(nonmask_cols) + 1, sharex='col', figsize=(12, 8))

    for row, column in enumerate([i for i in columns if 'mask' not in i]):

        ax[row].plot(df_epoch[time_column], df_epoch[column], label=column, color='black')

        if column == 'snr' and df_epoch_long is not None:
            ax[row].plot(df_epoch_long[time_column], df_epoch_long[column], color='dodgerblue')
        ax_labels[column] = row

    if 'wrist_nw_mask' in columns:
        wrist_ax = ax_labels[[i for i in ax_labels.keys() if 'wrist' in i][0]]
        ax[wrist_ax].fill_between(x=df_epoch[time_column], y1=0,
                                  y2=df_epoch['wrist_nw_mask']*df_epoch['wrist_avm'].max(),
                                  color='grey', alpha=.5, label='NW')

    if 'sleep_mask' in columns:
        wrist_ax = ax_labels[[i for i in ax_labels.keys() if 'wrist' in i][0]]
        ax[wrist_ax].fill_between(x=df_epoch[time_column], y1=0,
                                  y2=df_epoch['sleep_mask']*df_epoch['wrist_avm'].max()/2,
                                  color='purple', alpha=.33, label='Sleep')

    if 'sptw_mask' in columns:
        wrist_ax = ax_labels[[i for i in ax_labels.keys() if 'wrist' in i][0]]
        ax[wrist_ax].fill_between(x=df_epoch[time_column],
                                  y1=0,
                                  y2=df_epoch['sptw_mask']*df_epoch['wrist_avm'].max(),
                                  color='dodgerblue', alpha=.33, label='SPTW')

    if 'ankle_nw_mask' in columns:
        ankle_ax = ax_labels[[i for i in ax_labels.keys() if 'ankle' in i][0]]
        ax[ankle_ax].fill_between(x=df_epoch[time_column], y1=0,
                                  y2=df_epoch['ankle_nw_mask']*df_epoch['ankle_avm'].max(),
                                  color='grey', alpha=.5, label='NW')
    if 'gait_mask' in columns:
        ankle_ax = ax_labels[[i for i in ax_labels.keys() if 'ankle' in i][0]]
        ax[ankle_ax].fill_between(x=df_epoch[time_column], y1=0,
                                  y2=df_epoch['gait_mask']*df_epoch['ankle_avm'].max(),
                                  color='gold', alpha=.5, label='Gait')
    if 'chest_nw_mask' in columns:
        bitt_ax = ax_labels[[i for i in ax_labels.keys() if 'snr' in i][0]]

        ax[bitt_ax].fill_between(x=df_epoch[time_column], y1=0,
                                 y2=df_epoch['chest_nw_mask']*df_epoch['snr'].max(),
                                 color='grey', alpha=.5, label='NW')

    for row in ax:
        row.legend()

    if time_column == 'start_time':
        ax[-1].xaxis.set_major_formatter(xfmt)

    plt.tight_layout()

    return fig


def plotdensity_snr_by_type(df_epoch, group_col='wrist_intensity', ignore_nw=True, thresholds=(5, 18),
                            color_dict=None):

    df_use = df_epoch.loc[df_epoch['chest_nw_percent'] == 0] if ignore_nw else df_epoch
    groups = df_use.groupby(group_col)

    fig, ax = plt.subplots(1, figsize=(12, 8))

    if color_dict is not None:
        for group in color_dict.keys():
            groups['snr'].get_group(group).plot.density(ax=ax, color=color_dict[group], label=group)

    if color_dict is None:
        for group in groups.groups.keys():
            try:
                groups.get_group(group)['snr'].plot.density(ax=ax, label=group)
            except:
                pass

    x = ax.get_xlim()
    ax.axvspan(xmin=x[0], xmax=thresholds[0], ymin=0, ymax=1, color='red', alpha=.15, label='LowQ')
    ax.axvspan(xmin=thresholds[0], xmax=x[1], ymin=0, ymax=1, color='orange', alpha=.15, label='MedQ')
    ax.axvspan(xmin=thresholds[1], xmax=x[1], ymin=0, ymax=1, color='limegreen', alpha=.15, label='HighQ')

    ax.legend()
    ax.set_title(f"SNR by {group_col} (N={len(df_use['full_id'].unique())}; n={df_use.shape[0]})")
    ax.set_xlabel("SNR")
    ax.set_ylim(0, )

    return fig


def trend_heatmap(df, ignore_nw=True, use_full_id=True, include_avg=True):

    df = df.loc[~np.isnan(df['chest_nw_percent'])]

    if 'chest_nw_percent' in df.columns and ignore_nw:
        df = df.loc[df['chest_nw_percent'] == 0]

    p = pd.pivot(data=df, index="days", values='snr', columns="full_id").transpose()
    p.columns = [round(i, 5) for i in p.columns]

    val_range = [min([p[col].min() for col in p.columns]), max([p[col].max() for col in p.columns])]

    gridspec = [1]
    if include_avg:
        gridspec.append(1/p.shape[0])

    fig, ax = plt.subplots(1 if not include_avg else 2, figsize=(12, 8), sharex='col', gridspec_kw={'height_ratios': gridspec})
    use_ax = ax if not include_avg else ax[0]

    plt.subplots_adjust(right=.9)

    sns.heatmap(p, ax=use_ax, cmap='RdYlGn', vmin=val_range[0], vmax=val_range[1],
                cbar_ax=fig.add_axes([.925, .2, .05, .6]))

    if not include_avg:
        use_ax.set_xlabel("Days")

    use_ax.set_ylabel("")

    if not use_full_id:
        use_ax.set_yticklabels([f"#{i}" for i in np.arange(1, p.shape[0] + 1)])
        use_ax.set_ylabel("Participants")

    for i in np.arange(1, use_ax.get_ylim()[0]+1):
        use_ax.axhline(y=i, color='black', lw=2)

    if include_avg:
        p.loc['avg'] = [p[col].mean() for col in p.columns]
        sns.heatmap(p.iloc[-1:, :], ax=ax[1], cmap='RdYlGn', vmin=val_range[0], vmax=val_range[1], cbar=False)
        ax[1].set_ylabel("")

    last_ax = ax if not include_avg else ax[1]

    last_ax.set_xticks(np.arange(0, last_ax.get_xticks()[-1], 0.5))
    last_ax.set_xticklabels([round(i / 24, 3) for i in last_ax.get_xticks()])
    last_ax.set_xticks(np.arange(0, last_ax.get_xticks().max(), 24))
    last_ax.set_xticklabels([int(i) for i in np.arange(0, last_ax.get_xticks().max() / 24 + 1, 1)], rotation=0)
    last_ax.set_xlabel("Days")

    plt.suptitle(f"SNR Timeseries by Participant ({df.iloc[1]['hours']:.3f}-hour averaged)")
    plt.tight_layout()
    plt.subplots_adjust(right=.9)

    return p, fig


def plot_snr_cat_by_participant(df, ignore_nw=True, use_full_id=False, incl_average=True, bar_width=.75):

    if 'chest_nw_percent' in df.columns and ignore_nw:
        df = df.loc[df['chest_nw_percent'] == 0]

    fig, ax = plt.subplots(1, figsize=(12, 8))

    all_vals = pd.DataFrame()

    for subj_num, subj in enumerate(df['full_id'].unique()):
        df_subj = df.loc[df['full_id'] == subj]
        vals = df_subj['snr_cat'].value_counts()
        vals *= 100 / vals.sum()

        ax.barh([subj] if use_full_id else [f"#{subj_num+1}"], vals.loc['Q1'],
                color='limegreen', alpha=.75, edgecolor='black', height=bar_width)
        ax.barh([subj] if use_full_id else [f"#{subj_num+1}"], vals.loc['Q2'], left=vals.loc['Q1'],
                color='orange', alpha=.75, edgecolor='black', height=bar_width)
        ax.barh([subj] if use_full_id else [f"#{subj_num+1}"], vals.loc['Q3'], left=vals.loc['Q1'] + vals.loc['Q2'],
                color='red', alpha=.75, edgecolor='black', height=bar_width)

        vals = pd.DataFrame(vals.values, columns=[subj], index=['Q1', 'Q2', 'Q3']).transpose()
        all_vals = pd.concat([all_vals, vals])

    if incl_average:
        ax.barh(['average'], all_vals['Q1'].mean(),
                color='limegreen', alpha=.75, edgecolor='black', height=bar_width)
        ax.barh(['average'], all_vals['Q2'].mean(), left=all_vals['Q1'].mean(),
                color='orange', alpha=.75, edgecolor='black', height=bar_width)
        ax.barh(['average'], all_vals['Q3'].mean(), left=all_vals['Q1'].mean() + all_vals['Q2'].mean(),
                color='red', alpha=.75, edgecolor='black', height=bar_width)

    ax.set_xlim(0, 100)
    ax.set_ylim(ax.get_yticks()[0] - bar_width/2 - bar_width/10, ax.get_yticks()[-1] + bar_width/2 + bar_width/10)
    ax.set_xlabel("% of collection")
    ax.set_ylabel("Participants")
    ax.legend(['Q1', 'Q2', 'Q3'], bbox_to_anchor=[1, 1, 0, 0])

    plt.tight_layout()

    return all_vals, fig


def plot_wrist_ankle_snr(df_values):

    vals = np.arange(0, df_values['avm'].max(), 10)
    wrist_reg = np.polyfit(df_values[['avm', 'wrist_snr']].dropna()['avm'],
                           df_values[['avm', 'wrist_snr']].dropna()['wrist_snr'], deg=1)
    wrist_r = scipy.stats.pearsonr(df_values[['avm', 'wrist_snr']].dropna()['avm'],
                                   df_values[['avm', 'wrist_snr']].dropna()['wrist_snr'])[0]
    plt.plot(vals, [i * wrist_reg[0] + wrist_reg[1] for i in vals], color='black', label=f"r={wrist_r:.3f}",
             linestyle='dashed')
    plt.scatter(df_values['avm'], df_values['wrist_snr'], color='black', label='wrist')

    ankle_reg = np.polyfit(df_values[['avm', 'ankle_snr']].dropna()['avm'],
                           df_values[['avm', 'ankle_snr']].dropna()['ankle_snr'], deg=1)
    ankle_r = scipy.stats.pearsonr(df_values[['avm', 'ankle_snr']].dropna()['avm'],
                                   df_values[['avm', 'ankle_snr']].dropna()['ankle_snr'])[0]
    plt.plot(vals, [i * ankle_reg[0] + ankle_reg[1] for i in vals], color='dodgerblue', label=f"r={ankle_r:.3f}",
             linestyle='dashed')
    plt.scatter(df_values['avm'], df_values['ankle_snr'], color='dodgerblue', marker='v', label='ankle')

    plt.legend()
    plt.xlabel("AVM")
    plt.ylabel("SNR")
    plt.axhspan(xmin=vals[0], xmax=vals[-1], ymin=plt.ylim()[0], ymax=5, color='red', alpha=.15)
    plt.axhspan(xmin=vals[0], xmax=vals[-1], ymin=5, ymax=18, color='orange', alpha=.15)
    plt.axhspan(xmin=vals[0], xmax=vals[-1], ymin=18, ymax=plt.ylim()[1], color='limegreen', alpha=.15)


def avm_heatmap(df_desc, cmap='RdYlGn', norm_range=(-10, 30)):

    """ Cool colormaps:

    import matplotlib.colors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['red', 'orange', 'limegreen'])

    -darker with higher quality: 'binary'
    -sort of maps onto color coding of quality: 'RdYlGn'

    """

    fig, ax = plt.subplots(1, figsize=(10, 10))
    sns.heatmap(data=df_desc, ax=ax,
                cmap=cmap, vmin=norm_range[0], vmax=norm_range[1], center=(norm_range[1] - norm_range[0])/2,
                cbar_ax=fig.add_axes([.9, .2, .05, .6]))
    ax.set_xlabel("Wrist AVM")
    ax.set_ylabel("Ankle AVM")

    max_val = max([ax.get_yticks().max(), ax.get_xticks().max()])
    ticks = ax.get_yticks() if max(ax.get_yticks()) >= max(ax.get_xticks()) else ax.get_xticks()
    tick_labs = ax.get_yticklabels() if max(ax.get_yticks()) >= max(ax.get_xticks()) else ax.get_xticklabels()
    xticklab_rev = max(ax.get_yticks()) >= max(ax.get_xticks())

    ax.set_xticks(ticks)

    if xticklab_rev:
        ax.set_xticklabels([tick_labs[-(i+1)] for i in range(len(tick_labs))])
    if not xticklab_rev:
        ax.set_xticklabels(tick_labs)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labs)

    ax.set_xlim(0, max_val)
    ax.set_ylim(max_val, 0)

    plt.tight_layout()
    plt.subplots_adjust(right=.875)


def plot_nw_log(signal, acc_signal, ts, df_nw, df_nw_checked=None, downsample_ratio=3):

    fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))

    acc_ratio = int(round(len(signal) / len(acc_signal[0]), 0))
    ax[0].plot(ts[:min([len(signal), len(ts)]):downsample_ratio], signal[:min([len(ts), len(signal)]):downsample_ratio], color='black')

    ax[1].plot(ts[::acc_ratio][:min([len(ts[::acc_ratio]), len(acc_signal[0])])], acc_signal[0][:min([len(ts[::acc_ratio]), len(acc_signal[0])])], color='black')
    ax[1].plot(ts[::acc_ratio][:min([len(ts[::acc_ratio]), len(acc_signal[0])])], acc_signal[1][:min([len(ts[::acc_ratio]), len(acc_signal[0])])], color='red')
    ax[1].plot(ts[::acc_ratio][:min([len(ts[::acc_ratio]), len(acc_signal[0])])], acc_signal[2][:min([len(ts[::acc_ratio]), len(acc_signal[0])])], color='dodgerblue')

    if df_nw_checked is not None:
        for row in df_nw_checked.itertuples():
            ax[0].axvspan(xmin=row.time_removed, xmax=row.time_reattached, ymin=0, ymax=1,
                          color='orange', alpha=.35)

    for row in df_nw.itertuples():
        any_stamp = False
        try:
            ax[0].axvline(row.time_removed, color='red')
            any_stamp = True
        except TypeError:
            pass
        try:
            ax[0].axvline(row.time_reattached, color='limegreen')
            any_stamp = True
        except TypeError:
            pass

        if not any_stamp:
            print(f"CANNOT PLOT: #{[row.Index]} / {row.reason} || {row.time_removed} - {row.time_reattached}")

    ax[-1].xaxis.set_major_formatter(xfmt)


def scatter_snr_by_avm(df, min_datapoints=25, thresholds=(5, 18), use_sem=True):

    df_avm = df[['snr', 'wrist_avm_bin', 'ankle_avm_bin', 'chest_avm_bin']]
    N = len(df['full_id'].unique())

    w = df_avm.groupby("wrist_avm_bin").describe()['snr']
    a = df_avm.groupby("ankle_avm_bin").describe()['snr']
    c = df_avm.groupby("chest_avm_bin").describe()['snr']

    w = w.loc[w['count'] >= min_datapoints]
    a = a.loc[a['count'] >= min_datapoints]
    c = c.loc[c['count'] >= min_datapoints]

    wreg = np.polyfit(x=w.index, y=w['mean'], deg=1)
    wr = scipy.stats.pearsonr(w.index, w['mean'])
    areg = np.polyfit(x=a.index, y=a['mean'], deg=1)
    ar = scipy.stats.pearsonr(a.index, a['mean'])
    creg = np.polyfit(x=c.index, y=c['mean'], deg=1)
    cr = scipy.stats.pearsonr(c.index, c['mean'])

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.scatter(w.index, w['mean'], color='black', label='wrist', zorder=2, marker='o')
    ax.fill_between(x=w.index,
                    y1=w['mean'] - w['std'] / np.sqrt(w['count']) if use_sem else w['mean'] - w['std'],
                    y2=w['mean'] + w['std'] / np.sqrt(w['count']) if use_sem else w['mean'] + w['std'],
                    color='grey', alpha=.2)
    ax.plot(w.index, [i * wreg[0] + wreg[1] for i in w.index], color='black',
            linestyle='dashed', label=f"r2={wr[0]**2:.3f}", zorder=1)

    ax.scatter(a.index, a['mean'], color='orange', label='ankle', zorder=2, marker='v')
    ax.plot(a.index, [i * areg[0] + areg[1] for i in a.index], color='orange',
            linestyle='dashed', label=f"r2={ar[0]**2:.3f}", zorder=1)
    ax.fill_between(x=a.index,
                    y1=a['mean'] - a['std'] / np.sqrt(a['count']) if use_sem else a['mean'] - a['std'],
                    y2=a['mean'] + a['std'] / np.sqrt(a['count']) if use_sem else a['mean'] + a['std'],
                    color='orange', alpha=.2)

    ax.scatter(c.index, c['mean'], color='dodgerblue', label='chest', zorder=2, marker='x')
    ax.plot(c.index, [i * creg[0] + creg[1] for i in c.index], color='dodgerblue',
            linestyle='dashed', label=f"r2={cr[0]**2:.3f}", zorder=1)
    ax.fill_between(x=c.index,
                    y1=c['mean'] - c['std'] / np.sqrt(c['count']) if use_sem else c['mean'] - c['std'],
                    y2=c['mean'] + c['std'] / np.sqrt(c['count']) if use_sem else c['mean'] + c['std'],
                    color='dodgerblue', alpha=.2)

    ax.axhline(thresholds[0], color='red', lw=3, zorder=0)
    ax.axhline(thresholds[1], color='limegreen', lw=3, zorder=0)

    ax.legend()
    ax.set_xlabel("AVM")
    ax.set_ylabel("SNR")
    ax.set_xlim(-10, )

    ax.grid()
    ax.set_title(f"SNR by AVM; {N} participants")
    plt.tight_layout()

    d = pd.concat([w['mean'], a['mean'], c['mean']], axis=1)
    d.columns = ['wrist_snr', 'ankle_snr', 'chest_snr']

    df_stats = pd.DataFrame([list(wreg) + list(wr), list(areg) + list(ar), list(creg) + list(cr)],
                            columns=['m', 'b', 'r', 'p'],
                            index=['wrist', 'ankle', 'chest'])

    return d, df_stats, fig


def overall_sample():
    overall = np.random.normal(loc=16, scale=5, size=100)

    gait = np.random.normal(loc=12, scale=5, size=100)
    sleep = np.random.normal(loc=20, scale=2, size=100)

    supine = np.random.normal(loc=16, scale=5, size=100)
    prone = np.random.normal(loc=14, scale=5, size=100)
    upright_still = np.random.normal(loc=17, scale=7, size=100)
    upright_arm = np.random.normal(loc=14, scale=6, size=100)

    avm1 = np.random.normal(loc=22, scale=3, size=100)
    avm2 = np.random.normal(loc=19, scale=4, size=100)
    avm3 = np.random.normal(loc=17, scale=6, size=100)

    fig, ax = plt.subplots(1, figsize=(10, 6))

    c = ['black', 'limegreen', 'red', 'orange', 'gold', 'blue', 'lightgrey', 'dimgrey', 'black']
    # v = ['overall', 'gait', 'supine', 'prone', 'upright', 'sleep', 'avm1', 'avm2', 'avm3']
    # vars = [overall, gait, supine, prone, upright, sleep, avm1, avm2, avm3]
    vars = [overall, gait, supine, prone, upright_still, upright_arm]
    v = ['overall', 'gait', 'supine', 'prone', 'upright_still', 'upright_arm']
    for i, var in enumerate(vars):
        pd.Series(var).plot.density(ax=ax, color=c[i], label=v[i], linestyle='dashed' if v[i] == 'overall' else 'solid')
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel("Density")
    ax.axvspan(xmin=-15, xmax=5, ymin=0, ymax=1, color='red', zorder=0, alpha=.1)
    ax.axvspan(xmin=5, xmax=18, ymin=0, ymax=1, color='dodgerblue', zorder=0, alpha=.1)
    ax.axvspan(xmin=18, xmax=35, ymin=0, ymax=1, color='green', zorder=0, alpha=.1)
    ax.set_ylim(0, )
    ax.set_xlim(-5, 35)
    ax.legend()
    ax.set_title("Sample Data")
    plt.tight_layout()


def plot_density_by_posture(df_epoch: pd.DataFrame,
                            ignore_nw: bool = True,
                            colname: str or None = None,
                            thresholds: list or tuple = (5, 18),
                            xlim: list or tuple = (-5, 30)):
    """ Generates a density plot of SNR grouped by 'colname'

        Parameters
        ----------
        df_epoch
            dataframe with 'snr' and 'colname' columns
        ignore_nw
            if True, ignores nonwear
        colname
            column in df_epoch used to group SNR. If None, runs density plot without grouping
        thresholds
            SNR thresholds for Q1, Q2, Q3
        xlim
            plot x-axis limits

        Returns
        -------
         figure
    """

    fig, ax = plt.subplots(1, figsize=(10, 6))

    df = df_epoch.copy()

    if ignore_nw:
        df = df.loc[df['all_wear']]

    if colname is not None:

        df = df_epoch[df_epoch[colname].notna()]

        df_grouped = df.groupby(colname)

        c_dict = {"supine": 'brown', 'prone': 'lightgrey', 'sidelying': 'purple',
                  'upright_gait': 'dodgerblue', 'upright_active_nogait': 'red', 'upright_inactive': 'orange',
                  'other': 'pink'}
        for g in df_grouped.groups:
            df_grouped.get_group(g)['snr'].plot.density(ax=ax, label=g, color=c_dict[g])

    df['snr'].plot.density(ax=ax, label='all', color='black', linestyle='dashed')

    ax.legend()

    if xlim is None:
        xlim = ax.get_xlim()
    ax.axvspan(xmin=xlim[0], xmax=thresholds[0], ymin=0, ymax=1, color='red', alpha=.15)
    ax.axvspan(xmin=thresholds[0], xmax=thresholds[1], ymin=0, ymax=1, color='dodgerblue', alpha=.15)
    ax.axvspan(xmin=thresholds[1], xmax=xlim[1], ymin=0, ymax=1, color='limegreen', alpha=.15)

    ax.set_xlim(xlim)
    ax.set_ylim(0, )

    plt.tight_layout()

    return fig


def plot_snr_by_posture_barplot(df_epoch: pd.DataFrame,
                                ignore_nw: bool = True,
                                use_sem: bool = True):
    """ Generates barplot with SD bars of SNR for each posture

        Parameters
        ----------
        df_epoch
            dataframe with columns 'snr', 'posture_use', and 'all_wear'
        ignore_nw
            If True, ignores nonwear epochs
        use_sem:
            if True, error bars are standard error of mean. If False, standard deviation

        Returns
        -------
        figure
    """

    if ignore_nw:
        df = df_epoch.loc[df_epoch['all_wear']]
    if not ignore_nw:
        df = df_epoch.copy()

    df_g = df.groupby("posture_use")
    df_g_desc = df_g['snr'].describe()
    df_g_desc.loc['all'] = df['snr'].describe()

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    mult = np.sqrt(df_g_desc['count']) if use_sem else 1
    ax.bar(df_g_desc.index, df_g_desc['mean'], yerr=df_g_desc['std']/mult, capsize=4, color='grey', edgecolor='black')
    ax.set_ylabel("SNR")
    plt.tight_layout()

    return fig


def plot_avm_snr_scatter(df: pd.DataFrame,
                         ignore_nw: bool = True,
                         min_avm: int or float = 5):
    """ Generates a scatter plot for wrist, ankle, and chest AVM with respective regression lines

        Parameters
        ----------
        df
            df with columns 'any_nw', 'wrist_avm', 'ankle_avm', 'chest_avm', 'snr'
        ignore_nw
            if True, ignores nonwear epochs
        min_avm
            minimum AVM value. ignores values below this threshold

        Returns
        -------
        figure

    """

    if ignore_nw:
        df = df.loc[df['all_wear']]

    x = df['snr'].min(), df['snr'].max()
    df_wrist = df.loc[df['wrist_avm'] >= min_avm]
    df_ankle = df.loc[df['ankle_avm'] >= min_avm]
    df_chest = df.loc[df['chest_avm'] >= min_avm]

    wrist_reg = np.polyfit(df_wrist['snr'], df_wrist['wrist_avm'], deg=1)
    wrist_r = scipy.stats.pearsonr(df_wrist['snr'], df_wrist['wrist_avm'])[0]

    ankle_reg = np.polyfit(df_ankle['snr'], df_ankle['ankle_avm'], deg=1)
    ankle_r = scipy.stats.pearsonr(df_ankle['snr'], df_ankle['ankle_avm'])[0]

    chest_reg = np.polyfit(df_chest['snr'], df_chest['chest_avm'], deg=1)
    chest_r = scipy.stats.pearsonr(df_chest['snr'], df_chest['chest_avm'])[0]

    fig, ax = plt.subplots(1, figsize=(8, 8))
    plt.suptitle(f"AVM threshold = {min_avm}")

    ax.scatter(df_wrist['snr'], df_wrist['wrist_avm'], color='black',
               label=f'wrist (r2 = {wrist_r**2:.3f}, n = {df_wrist.shape[0]})')
    ax.plot(x, [wrist_reg[0] * i + wrist_reg[1] for i in x], color='black', linestyle='dashed')

    ax.scatter(df_chest['snr'], df_chest['ankle_avm'], color='red',
               label=f'ankle (r2 = {ankle_r**2:.3f}, n = {df_ankle.shape[0]})', alpha=.5)
    ax.plot(x, [ankle_reg[0] * i + ankle_reg[1] for i in x], color='red', linestyle='dashed')

    ax.scatter(df_chest['snr'], df_chest['chest_avm'], color='dodgerblue',
               label=f'chest (r2 = {chest_r**2:.3f}, n = {df_chest.shape[0]})', alpha=.5)
    ax.plot(x, [chest_reg[0] * i + chest_reg[1] for i in x], color='dodgerblue', linestyle='dashed')

    ax.axhspan(xmin=0, xmax=1, ymin=ax.get_ylim()[0], ymax=min_avm, alpha=.25,
               color='grey', label='below AVM thresh.')

    ax.legend()

    ax.set_ylabel("AVM")
    ax.set_xlabel("SNR")

    return fig


def pieplot_snr_categories(df: pd.DataFrame,
                           ignore_nw: bool = True):
    """ Creates pie plot of SNR qualities in given df

        Parameters
        ----------
        df
            dataframe of data. requires 'snr_quality' column.
            If ignore_nw == True, also requires 'chest_nw_mask' or 'chest_nw_percent' column
        ignore_nw
            if True, checks for chest nonwear column and ignores rows of data that indicate nonwear

        Returns
        -------
        figure
    """

    df = df.copy()

    if ignore_nw:
        if 'chest_nw_mask' in df.columns:
            df = df.loc[df['chest_nw_mask'] == 0]
        if 'chest_nw_percent' in df.columns:
            df = df.loc[df['chest_nw_percent'] == 0]

    vals = df['snr_quality'].value_counts() / df.shape[0] * 100

    def fmt_percent(values):
        out = []
        for key in values.keys():
            out.append(f"{key} ({values[key]:.1f}" + "%)")

        return out

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.pie(x=vals, colors=['limegreen', 'dodgerblue', 'red'], wedgeprops={"linewidth": 1, 'edgecolor': 'black'},
           labels=fmt_percent(vals))

    return fig


def plot_subject_alldata(subj_dict):

    fig, ax = plt.subplots(6, sharex='col', figsize=(12, 9.5), gridspec_kw={'height_ratios': [1, 1, 1, 1, .5, .5]})

    ax[0].plot(subj_dict['df1s']['start_time'], subj_dict['df1s']['wrist_avm'], color='dodgerblue', label='1s')
    ax[0].plot(subj_dict['epoch_med']['start_time'], subj_dict['epoch_med']['wrist_avm'], color='red', label='5s')
    ax[0].plot(subj_dict['epoch_long']['start_time'], subj_dict['epoch_long']['wrist_avm'], color='black', label='900s')
    ax[0].set_ylabel("Wrist AVM")
    ax[0].legend(loc='lower right')
    ax[0].set_ylim(0, )

    ax[1].plot(subj_dict['df1s']['start_time'], subj_dict['df1s']['chest_avm'], color='dodgerblue', label='1s')
    ax[1].plot(subj_dict['epoch_med']['start_time'], subj_dict['epoch_med']['chest_avm'], color='red', label='5s')
    ax[1].plot(subj_dict['epoch_long']['start_time'], subj_dict['epoch_long']['chest_avm'], color='black', label='900s')
    ax[1].set_ylabel("Chest AVM")

    for row in subj_dict['chest_weber_nw'].itertuples():
        ax[1].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1, color='red', alpha=.25)
        ax[3].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1, color='red', alpha=.25)

    ax[1].legend(loc='lower right')
    ax[1].set_ylim(0, )

    ax[2].plot(subj_dict['df1s']['start_time'], subj_dict['df1s']['ankle_avm'], color='dodgerblue', label='1s')
    ax[2].plot(subj_dict['epoch_med']['start_time'], subj_dict['epoch_med']['ankle_avm'], color='red', label='5s')
    ax[2].plot(subj_dict['epoch_long']['start_time'], subj_dict['epoch_long']['ankle_avm'], color='black', label='900s')
    ax[2].set_ylabel("Ankle AVM")
    ax[2].legend(loc='lower right')
    ax[2].set_ylim(0, )

    for ax_i, col in enumerate(['wrist_nw', 'chest_nw', 'ankle_nw']):
        for row in subj_dict[col].itertuples():
            ax[ax_i].axvspan(xmin=row.start_time, xmax=row.end_time, color='grey', alpha=.25)

            if ax_i == 1:
                ax[3].axvspan(xmin=row.start_time, xmax=row.end_time, color='grey', alpha=.25)

    ax[3].plot(subj_dict['df1s']['start_time'], subj_dict['df1s']['snr'], color='dodgerblue', label='1s')
    ax[3].plot(subj_dict['epoch_med']['start_time'], subj_dict['epoch_med']['snr'], color='red', label='5s')
    ax[3].plot(subj_dict['epoch_long']['start_time'], subj_dict['epoch_long']['snr'], color='black', label='900s')
    ax[3].set_ylabel("SNR")
    ax[3].legend(loc='lower right')

    ax[4].plot(subj_dict['df1s']['start_time'], subj_dict['df1s']['sleep_mask'], color='dodgerblue', label='1s')
    ax[4].plot(subj_dict['epoch_med']['start_time'], subj_dict['epoch_med']['sleep_any'], color='red', label='5s')
    ax[4].plot(subj_dict['epoch_long']['start_time'], subj_dict['epoch_long']['sleep_any'], color='black', label='900s')
    ax[4].set_ylabel("Sleep")
    ax[4].set_yticks([0, 1])
    ax[4].set_yticklabels(['wake', 'sleep'])
    ax[4].legend(loc='lower right')

    ax[5].plot(subj_dict['df1s']['start_time'], subj_dict['df1s']['gait_mask'], color='dodgerblue', label='1s')
    ax[5].plot(subj_dict['epoch_med']['start_time'], subj_dict['epoch_med']['gait_any'], color='red', label='5s')
    ax[5].plot(subj_dict['epoch_long']['start_time'], subj_dict['epoch_long']['gait_any'], color='black', label='900s')
    ax[5].set_ylabel("Gait")
    ax[5].set_yticks([0, 1])
    ax[5].set_yticklabels(['no', 'gait'])
    ax[5].legend(loc='lower right')

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()
    plt.subplots_adjust(hspace=.1)

    return fig
