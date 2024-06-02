import os
from multiprocessing import Pool

import neurokit2
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

time_middle_file_path = 'middle-files/'
hrv_output_path = 'hrv-middle-data/'
test_minutes = 2

def hrv_extraction(p_index):
    participant_key = 'P' + str(p_index + 1).zfill(2)

    df_time = pd.read_csv(f'{time_middle_file_path}{participant_key}_app_usage_events_with_confidences.csv')
    df_hrv = pd.read_csv(f'dataset/{participant_key}/RRI.csv')

    df_analyse = pd.DataFrame()
    df_analyse['RRI'] = df_hrv.interval
    df_analyse['RRI_Time_ms'] = df_hrv.timestamp
    df_analyse['RRI_Time'] = df_hrv.timestamp / 1000

    # premise of the hrv extracting data function
    df_analyse.sort_values(by='RRI_Time_ms')

    # filter out use time that is less than 1 minute
    filtered_use_time = df_time.loc[df_time.time_difference > test_minutes * 60 * 1000]

    loc_time = filtered_use_time.loc[df_time['confidenceStill'] > 0.7]

    res = None
    for index, row in loc_time.iterrows():
        foreground_time = row['foreground_time']
        background_time = row['background_time']

        df_slice = df_analyse.loc[(df_analyse['RRI_Time_ms'] >= foreground_time) & (df_analyse['RRI_Time_ms'] <= background_time)]

        # there might be no data in RRI.csv, maybe participants didn't wear the band in these cases.
        # 120: roughly 2 minutes
        if len(df_slice) < 120:
            print(participant_key + ' empty slice at foreground time: ' + str(foreground_time))
            continue

        res_row = neurokit2.hrv_time(df_slice)

        res_row['foreground_time'] = foreground_time
        res_row['background_time'] = background_time

        if res is None:
            res = res_row
        else:
            res = pd.concat([res, res_row], ignore_index=True)

    if res is not None:
        res.to_csv(hrv_output_path + participant_key + '_hrv_result.csv', index=False)
        # filtered_use_time.to_csv(hrv_output_path + participant_key + 'time_before_confidence_filter.csv', index=False)
        # loc_time.to_csv(hrv_output_path + participant_key + 'time_after_filter.csv', index=False)
    else:
        print(participant_key + ' has no result')
    print(participant_key + ' done')

def process_hrv_async():
    # left one core for me to use my computer...
    p = Pool()
    for i in range(80):
        p.apply_async(hrv_extraction, args=(i,))
    p.close()
    p.join()


if __name__ == '__main__':
    # process_hrv_async()
    hrv_extraction(0)
