import os
from multiprocessing import Pool

import neurokit2
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

time_middle_file_path = 'middle-files/'
hrv_output_path = 'hrv-middle-data/'


def hrv_extraction(p_index):
    participant_key = 'P' + str(p_index + 1).zfill(2)

    df_time = pd.read_csv(f'{time_middle_file_path}{participant_key}_foreground_background_differences.csv')
    df_hrv = pd.read_csv(f'dataset/{participant_key}/RRI.csv')

    df_analyse = pd.DataFrame()
    df_analyse['RRI'] = df_hrv.interval
    df_analyse['RRI_Time_ms'] = df_hrv.timestamp
    df_analyse['RRI_Time'] = df_hrv.timestamp / 1000

    # premise of the hrv extracting data function
    df_analyse.sort_values(by='RRI_Time_ms')

    # filter out use time that is less than 1 minute
    loc_time = df_time.loc[df_time.time_difference > 60 * 1000]

    res = None
    for index, row in loc_time.iterrows():
        foreground_time = row['foreground_time']
        background_time = row['background_time']

        (ts_before, ts_target, ts_after) = get_key_timestamps(foreground_time, 60 * 1000, df_analyse,
                                                              df_key='RRI_Time_ms')

        if ts_before > 0 and ts_target > 0 and ts_after > 0:
            before_res = neurokit2.hrv_time(
                df_analyse.loc[(df_analyse['RRI_Time_ms'] >= ts_before) & (df_analyse['RRI_Time_ms'] <= ts_target)]
            )

            after_res = neurokit2.hrv_time(
                df_analyse.loc[(df_analyse['RRI_Time_ms'] > ts_target) & (df_analyse['RRI_Time_ms'] <= ts_after)]
            )

            res_row = pd.merge(before_res, after_res, suffixes=('_before', '_after'), left_index=True, right_index=True)
            res_row['foreground_time'] = foreground_time

            if res is None:
                res = res_row
            else:
                res = pd.concat([res, res_row], ignore_index=True)

    res.to_csv(hrv_output_path + participant_key + '_hrv_result.csv', index=False)
    print(participant_key + ' done')


def get_key_timestamps(target, duration_ms, df, df_key='timestamp'):
    target_before = target - duration_ms
    target_after = target + duration_ms

    timestamp_before = 0
    timestamp_target = 0
    timestamp_after = 0

    # premise: df[df_key] is sorted
    for df_index, df_row in df.iterrows():
        if df_row[df_key] <= target_before:
            timestamp_before = df_row[df_key]
        elif df_row[df_key] <= target:
            timestamp_target = df_row[df_key]
        elif df_row[df_key] >= target_after:
            timestamp_after = df_row[df_key]
            break
    return timestamp_before, timestamp_target, timestamp_after


def process_hrv_async():
    # left one core for me to use my computer...
    p = Pool(os.cpu_count() - 1)
    for i in range(80):
        p.apply_async(hrv_extraction, args=(i,))
    p.close()
    p.join()


if __name__ == '__main__':
    process_hrv_async()

