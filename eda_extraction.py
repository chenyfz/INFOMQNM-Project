import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from neurokit2 import eda_process

plt.style.use('ggplot')

time_middle_file_path = 'middle-files/'
eda_output_path = 'eda-middle-data/'


def eda_extraction(p_index):
    participant_key = 'P' + str(p_index + 1).zfill(2)

    df_time = pd.read_csv(f'{time_middle_file_path}{participant_key}_foreground_background_differences.csv')
    df_eda = pd.read_csv(f'dataset/{participant_key}/EDA.csv')
    df_eda = df_eda.loc[df_eda['resistance'] != 0]

    # transform from ohm to microsiemens (Note: the original paper said it is kilo-ohm but seems wrong...)
    df_eda['conductance'] = 1 / df_eda['resistance'] * 1000

    # potential: transform to z-score?
    df_eda['conductance'] = stats.zscore(df_eda['conductance'])

    # filter out use time that is less than 1 minute
    loc_time = df_time.loc[df_time.time_difference > 60 * 1000]

    res = pd.DataFrame(columns=['foreground_time', 'max_amplitude'])
    for index, row in loc_time.iterrows():
        foreground_time = row['foreground_time']
        background_time = row['background_time']

        (ts_target, ts_after) = get_key_timestamps(foreground_time, 60 * 1000, df_eda)

        if ts_target > 0 and ts_after > 0:
            df_analyse = df_eda.loc[(df_eda['timestamp'] >= ts_target) & (df_eda['timestamp'] <= ts_after)]

            # data is too short to get meaningful result
            if (len(df_analyse)) < 15 * 2:
                continue

            eda_res, info = eda_process(df_analyse['conductance'], 5, kwargs_phasic='SparsEDA')

            res.loc[len(res)] = [foreground_time, np.nanmax(info['SCR_Amplitude'])]

    res.to_csv(eda_output_path + participant_key + '_eda_result.csv', index=False)
    print(participant_key + ' done')


def get_key_timestamps(target, duration_ms, df, df_key='timestamp'):
    target_after = target + duration_ms

    idx_target = df[df_key].searchsorted(target, 'right') - 1
    idx_after = df[df_key].searchsorted(target_after, 'left')

    timestamp_target = df[df_key].iloc[idx_target] if idx_target >= 0 else 0
    timestamp_after = df[df_key].iloc[idx_after] if idx_after < len(df) else 0

    return timestamp_target, timestamp_after


def process_eda_async():
    # left one core for me to use my computer...
    p = Pool(os.cpu_count() - 1)
    for i in range(80):
        p.apply_async(eda_extraction, args=(i,))
    p.close()
    p.join()


if __name__ == '__main__':
    process_eda_async()
    # eda_extraction(8)
