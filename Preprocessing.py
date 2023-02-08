import os
import nimbalwear
from nimbalwear.activity import activity_wrist_avm
import pandas as pd
from tqdm import tqdm


def process_avm(full_id,
                edf_folder='W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/',
                epoch_data=False,
                epoch_len=1,
                savedir="",
                save_file=False,
                device_loc="Wrist"):

    d = None

    if device_loc in ['chest', 'Chest']:
        fname = f"{edf_folder}{full_id}_01_BF36_Chest.edf"

    if device_loc in ['ankle', 'Ankle', 'Wrist', 'wrist']:
        fname = f"{edf_folder}{full_id}_01_AXV6_R{device_loc}.edf"

        if not os.path.exists(fname):
            fname = f"{edf_folder}{full_id}_01_AXV6_L{device_loc}.edf"

    if not os.path.exists(fname):
        print(f"File {fname} not found")
        return None, None

    if os.path.exists(fname):
        d = nimbalwear.Device()
        d.import_edf(fname)
        d.autocal()

        if epoch_data:
            start_key = "start_datetime" if "start_datetime" in d.header.keys() else 'startdate'
            df_avm = activity_wrist_avm(x=d.signals[d.get_signal_index("Accelerometer x")],
                                        y=d.signals[d.get_signal_index("Accelerometer y")],
                                        z=d.signals[d.get_signal_index("Accelerometer z")],
                                        sample_rate=d.signal_headers[d.get_signal_index("Accelerometer x")]['sample_rate'],
                                        start_datetime=d.header[start_key], epoch_length=epoch_len,
                                        lowpass=12 if d.signal_headers[d.get_signal_index("Accelerometer x")]['sample_rate'] <= 41 else 20)[0]
            df_avm = df_avm[["start_time", "end_time", 'avm']]
            df_avm['full_id'] = [full_id] * df_avm.shape[0]

            if save_file:
                df_avm.to_csv(f"{savedir}{full_id}_01_{device_loc}_AVM.csv", index=False)
                print(f"-{device_loc} AVM file saved to {savedir}")

    return d, df_avm


def create_avm_file(full_id: str,
                    edf_folder: str = 'W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/',
                    savedir: str = "O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/Data/dev_data/"):

    print("\nCreating ankle, wrist, and chest accelerometer 1-second AVM files...")

    for device in tqdm(['Ankle', 'Wrist', 'Chest']):

        d, df_avm = process_avm(full_id=full_id,
                                edf_folder=edf_folder,
                                epoch_data=True,
                                epoch_len=1,
                                save_file=True,
                                device_loc=device,
                                savedir=savedir + f"{device.lower()}_avm/")


def combine_df_avm(full_id,
                   wrist_path="O:/OBI/ONDRI@Home/Papers/ECG Signal Quality/Data/dev_data/wrist_avm/",
                   ankle_path="O:/OBI/ONDRI@Home/Papers/ECG Signal Quality/Data/dev_data/ankle_avm/",
                   chest_path="O:/OBI/ONDRI@Home/Papers/ECG Signal Quality/Data/dev_data/chest_avm/",
                   save_dir="O:/OBI/ONDRI@Home/Papers/ECG Signal Quality/Data/dev_data/avm/",
                   save_file=False):

    print(f"\nCombining wrist, ankle, and chest AVM files for {full_id}...")
    w_file = f"{wrist_path}{full_id}_01_Wrist_AVM.csv"
    if os.path.exists(w_file):
        w = pd.read_csv(w_file)
    if not os.path.exists(w_file):
        print(f"    -{w_file} does not exist")

    a_file = f"{ankle_path}{full_id}_01_Ankle_AVM.csv"
    if os.path.exists(a_file):
        a = pd.read_csv(a_file)
    if not os.path.exists(a_file):
        print(f"    -{a_file} does not exist")

    c_file = f"{chest_path}{full_id}_01_Chest_AVM.csv"
    if os.path.exists(c_file):
        c = pd.read_csv(c_file)
    if not os.path.exists(c_file):
        print(f"    -{c_file} does not exist")

    start = max([w['start_time'].iloc[0], a['start_time'].iloc[0], c['start_time'].iloc[0]])
    end = min([w['start_time'].iloc[-1], a['start_time'].iloc[-1], c['start_time'].iloc[-1]])

    w = w.loc[(w['start_time'] >= start) & (w['start_time'] <= end)].reset_index(drop=True)
    a = a.loc[(a['start_time'] >= start) & (a['start_time'] <= end)].reset_index(drop=True)
    c = c.loc[(c['start_time'] >= start) & (c['start_time'] <= end)].reset_index(drop=True)

    df_comb = pd.DataFrame({"start_time": w['start_time'], 'wrist_avm': w['avm'],
                            'ankle_avm': a['avm'], "chest_avm": c['avm']})

    if save_file:
        df_comb.to_csv(f"{save_dir}{full_id}_avm.csv", index=False)
        print(f"-Saved to {save_dir}")

    return df_comb
