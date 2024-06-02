import glob
import os
from multiprocessing import Pool

import neurokit2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('ggplot')

time_middle_file_path = 'middle-files/'
hrv_output_path = 'hrv-middle-data/'
eda_output_path = 'eda-middle-data/'
minimal_duration_minutes = 2
confidence_still_threshold = 0.7

def hrv_extraction(p_index):
    participant_key = 'P' + str(p_index + 1).zfill(2)

    df_time = pd.read_csv(f'{time_middle_file_path}{participant_key}_app_usage_events_with_confidences.csv')
    df_hrv = pd.read_csv(f'dataset/{participant_key}/RRI.csv')
    df_hrv.sort_values(by='timestamp')

    df_hrv['RRI'] = df_hrv['interval']
    df_hrv['RRI_Time_ms'] = df_hrv['timestamp']
    df_hrv['RRI_Time'] = df_hrv['timestamp'] / 1000

    filtered_use_time = df_time.loc[df_time.time_difference > minimal_duration_minutes * 60 * 1000]
    loc_time = filtered_use_time.loc[df_time['confidenceStill'] >= confidence_still_threshold]

    res = None
    for index, row in loc_time.iterrows():
        foreground_time = row['foreground_time']
        background_time = row['background_time']

        df_analyse = df_hrv.loc[
            (df_hrv['timestamp'] >= foreground_time) & (df_hrv['timestamp'] <= background_time)]

        if len(df_analyse) == 0:
            continue

        # participants maybe not wearing the band in some cases, so check again here.
        if df_analyse.iloc[-1]['timestamp'] - df_analyse.iloc[0]['timestamp'] < minimal_duration_minutes * 60 * 1000:
            continue

        res_row = neurokit2.hrv_time(df_analyse)
        res_row['foreground_time'] = foreground_time
        res_row['background_time'] = background_time

        if res is None:
            res = res_row
        else:
            res = pd.concat([res, res_row], ignore_index=True)

    if res is not None:
        res.to_csv(hrv_output_path + participant_key + '_hrv_result.csv', index=False)
        print(f'[HRV]: {participant_key} extraction done')
    else:
        print(f'[HRV]: {participant_key} has no result')


def eda_extraction(p_index):
    participant_key = 'P' + str(p_index + 1).zfill(2)

    df_time = pd.read_csv(f'{time_middle_file_path}{participant_key}_app_usage_events_with_confidences.csv')
    df_eda = pd.read_csv(f'dataset/{participant_key}/EDA.csv')
    df_eda = df_eda.loc[df_eda['resistance'] != 0]

    # transform from ohm to microsiemens (Note: the original paper said it is kilo-ohm but seems wrong...)
    df_eda['conductance'] = 1 / df_eda['resistance'] * 1000

    # filter out use time that is less than 1 minute
    filtered_use_time = df_time.loc[df_time.time_difference > minimal_duration_minutes * 60 * 1000]

    loc_time = filtered_use_time.loc[df_time['confidenceStill'] > confidence_still_threshold]

    res = pd.DataFrame(columns=['foreground_time', 'max_amplitude', 'scr_count_per_minute'])
    for index, row in loc_time.iterrows():
        foreground_time = row['foreground_time']
        background_time = row['background_time']

        df_analyse = df_eda.loc[(df_eda['timestamp'] >= foreground_time) & (df_eda['timestamp'] <= background_time)]

        # data is too short to get meaningful result
        if (len(df_analyse)) < 15 * 2:
            continue

        time_diff_minutes = df_analyse.iloc[-1]['timestamp'] - df_analyse.iloc[0]['timestamp'] / 1000 / 60

        eda_res, info = neurokit2.eda_process(df_analyse['conductance'], 5, kwargs_phasic='SparsEDA')

        res.loc[len(res)] = [
            foreground_time,
            np.nanmax(eda_res['SCR_Amplitude']),
            len(info) / time_diff_minutes,
        ]

    if len(res) > 0:
        res.to_csv(eda_output_path + participant_key + '_eda_result.csv', index=False)
        print(f'[EDA]: {participant_key} extraction done')
    else:
        print(f'[EDA]: {participant_key} has no result')


def process_hrv_async():
    # left one core for me to use my computer...
    p = Pool()
    for i in range(80):
        p.apply_async(hrv_extraction, args=(i,))
    p.close()
    p.join()


def process_eda_async():
    # left one core for me to use my computer...
    p = Pool(os.cpu_count() - 1)
    for i in range(80):
        p.apply_async(eda_extraction, args=(i,))
    p.close()
    p.join()


def clear_hrv_middle_files():
    files = glob.glob(hrv_output_path + '*')
    for f in files:
        os.remove(f)


def clear_eda_middle_files():
    files = glob.glob(eda_output_path + '*')
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    Path(hrv_output_path).mkdir(parents=True, exist_ok=True)
    clear_hrv_middle_files()
    process_hrv_async()
    # hrv_extraction(8)

    Path(eda_output_path).mkdir(parents=True, exist_ok=True)
    clear_eda_middle_files()
    process_eda_async()
