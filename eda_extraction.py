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
test_minutes = 2


def eda_extraction(p_index):
    participant_key = 'P' + str(p_index + 1).zfill(2)

    df_time = pd.read_csv(f'{time_middle_file_path}{participant_key}_app_usage_events_with_confidences.csv')
    df_eda = pd.read_csv(f'dataset/{participant_key}/EDA.csv')
    df_eda = df_eda.loc[df_eda['resistance'] != 0]

    # transform from ohm to microsiemens (Note: the original paper said it is kilo-ohm but seems wrong...)
    df_eda['conductance'] = 1 / df_eda['resistance'] * 1000

    # potential: transform to z-score?
    df_eda['conductance'] = stats.zscore(df_eda['conductance'])

    # filter out use time that is less than 1 minute
    filtered_use_time = df_time.loc[df_time.time_difference > test_minutes * 60 * 1000]

    loc_time = filtered_use_time.loc[df_time['confidenceStill'] > 0.7]

    res = pd.DataFrame(columns=['foreground_time', 'max_amplitude', 'scr_count_per_minute'])
    for index, row in loc_time.iterrows():
        foreground_time = row['foreground_time']
        background_time = row['background_time']
        time_diff_minutes = row['time_difference'] / 1000 / 60

        df_analyse = df_eda.loc[(df_eda['timestamp'] >= foreground_time) & (df_eda['timestamp'] <= background_time)]

        # data is too short to get meaningful result
        if (len(df_analyse)) < 15 * 2:
            continue

        eda_res, info = eda_process(df_analyse['conductance'], 5, kwargs_phasic='SparsEDA')

        res.loc[len(res)] = [
            foreground_time,
            np.nanmax(info['SCR_Amplitude']),
            np.count_nonzero(~np.isnan(info['SCR_Amplitude'])) / time_diff_minutes,
        ]

    res.to_csv(eda_output_path + participant_key + '_eda_result.csv', index=False)
    print(participant_key + ' done')

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
