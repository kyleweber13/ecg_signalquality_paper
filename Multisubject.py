import pandas as pd
import os
from tqdm import tqdm


def combine_dfs(files: list or tuple = ()):

    df = pd.read_csv(files[0]) if "csv" in files[0] else pd.read_excel(files[0])
    df['full_id'] = [os.path.basename(files[0]).split("_")[0] + "_" + os.path.basename(files[0]).split("_")[1]] * df.shape[0]

    if len(files) >= 2:
        for file in tqdm(files):
            if file == files[0]:
                pass
            else:
                df2 = pd.read_csv(files[0]) if "csv" in files[0] else pd.read_excel(files[0])
                df2['full_id'] = [os.path.basename(file).split("_")[0] + "_" + os.path.basename(file).split("_")[1]] * df2.shape[0]

                df = pd.concat([df, df2])

    return df


paper_dir = 'O:/OBI/ONDRI@Home/Papers/Kyle - ECG Signal Quality/'
df_1s_filenames = [f"{paper_dir}Data/dev_data/processed_epoch1s/" + i for i in os.listdir(f"{paper_dir}Data/dev_data/processed_epoch1s/")]

df_all = combine_dfs(df_1s_filenames)
print(df_all['full_id'].value_counts())