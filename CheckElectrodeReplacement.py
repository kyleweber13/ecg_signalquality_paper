import matplotlib.units
import nimbalwear
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
import pandas as pd
import os
import numpy as np
from Filtering import filter_signal

full_id = 'OND09_SBH0156'  # 0088, 0152
edf_folder = 'W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/'
ecg_fname = f"{edf_folder}{full_id}_01_BF36_Chest.edf"

df_removal = pd.read_excel("O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/electrode_removals.xlsx")
df_removal = df_removal.loc[df_removal['full_id'] == full_id]

nw_file = f"C:/Users/ksweber/Desktop/ECG_nonwear_dev/FinalBouts_NoSNR/{full_id}_01_BF36_Chest_NONWEAR.csv"
if os.path.exists(nw_file):
    df_nw = pd.read_csv(nw_file)
if not os.path.exists(nw_file):
    df_nw = pd.DataFrame(columns=['start_time', 'end_time'])

ecg = nimbalwear.Device()
ecg.import_edf(ecg_fname)
ecg.ts = pd.date_range(start=ecg.header['start_datetime'], periods=len(ecg.signals[0]), freq=f"{1000/ecg.signal_headers[0]['sample_rate']}ms")
ecg.filt = filter_signal(data=ecg.signals[0], sample_f=ecg.signal_headers[0]['sample_rate'], low_f=1, high_f=40, filter_order=5, filter_type='bandpass')

fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))
ax[0].plot(ecg.ts[::2], ecg.signals[0][::2], color='red')
ax[0].plot(ecg.ts[::2], ecg.filt[::2], color='black')

ax[1].plot(ecg.ts[::20], ecg.signals[1][::2], color='black')
ax[1].plot(ecg.ts[::20], ecg.signals[2][::2], color='red')
ax[1].plot(ecg.ts[::20], ecg.signals[3][::2], color='dodgerblue')
ax[2].plot(ecg.ts[::250], ecg.signals[ecg.get_signal_index('Temperature')], color='red')

for i in [df_removal['removal_time'].iloc[0]]:
    for a in range(3):
        ax[a].axvline(x=pd.to_datetime(i), color='orange', lw=3, linestyle='dashed')
    print(df_removal['notes'].iloc[0])

for row in df_nw.itertuples():
    for a in range(3):
        ax[a].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1, color='grey', alpha=.15)
plt.tight_layout()

try:
    for a in range(3):
        ax[a].axvspan(xmin=df_removal['crop_time'].iloc[0], xmax=ecg.ts[-1], ymin=0, ymax=1, color='red', alpha=.1)
except matplotlib.units.ConversionError:
    pass

ax[-1].xaxis.set_major_formatter(xfmt)
