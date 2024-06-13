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
confidence_active_threshold = 0.25

def hrv_extraction(p_index):
    participant_key = 'P' + str(p_index + 1).zfill(2)

    # read the results of timeframe_extraction.py
    df_time = pd.read_csv(f'{time_middle_file_path}{participant_key}_app_usage_events_with_confidences.csv')
    df_hrv = pd.read_csv(f'dataset/{participant_key}/RRI.csv')
    df_hrv.sort_values(by='timestamp')

    # column names required by NeuroKit 2
    df_hrv['RRI'] = df_hrv['interval']
    df_hrv['RRI_Time_ms'] = df_hrv['timestamp']
    df_hrv['RRI_Time'] = df_hrv['timestamp'] / 1000

    # filter out timeframes with duration less than the defined one
    filtered_use_time = df_time.loc[df_time.time_difference > minimal_duration_minutes * 60 * 1000]

    # filter out timeframes that participants have confidence level of in active higher than a defined threshold
    loc_time = filtered_use_time.loc[df_time['confidenceOnFoot'] + df_time['confidenceOnBicycle']
                                     <= confidence_active_threshold]

    # feature extraction
    res = None
    for index, row in loc_time.iterrows():
        foreground_time = row['foreground_time']
        background_time = row['background_time']

        df_analyse = df_hrv.loc[
            (df_hrv['timestamp'] >= foreground_time) & (df_hrv['timestamp'] <= background_time)]

        # using smartphone while not wearing MS Band 2
        if len(df_analyse) == 0:
            continue
        # double check filter: timeframes with duration less than the defined one
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

    # read the results of timeframe_extraction.py
    df_time = pd.read_csv(f'{time_middle_file_path}{participant_key}_app_usage_events_with_confidences.csv')
    df_eda = pd.read_csv(f'dataset/{participant_key}/EDA.csv')

    # resistance will be divided later, a good programmer always check division by zero problem!
    df_eda = df_eda.loc[df_eda['resistance'] != 0]

    # transform from kilo-ohm to microsiemens
    df_eda['conductance'] = 1000 / df_eda['resistance']

    # filter out timeframes with duration less than the defined one
    filtered_use_time = df_time.loc[df_time.time_difference > minimal_duration_minutes * 60 * 1000]

    # filter out timeframes that participants have confidence level of in active higher than a defined threshold
    loc_time = filtered_use_time.loc[df_time['confidenceOnFoot'] + df_time['confidenceOnBicycle']
                                     <= confidence_active_threshold]

    res = pd.DataFrame(columns=['foreground_time', 'max_amplitude', 'scr_count_per_minute'])
    for index, row in loc_time.iterrows():
        foreground_time = row['foreground_time']
        background_time = row['background_time']

        df_analyse = df_eda.loc[(df_eda['timestamp'] >= foreground_time) & (df_eda['timestamp'] <= background_time)]

        # double check if data is too short to get meaningful result
        if (len(df_analyse)) < 15 * 2:
            continue

        # accurate minute durations based on the row of datas
        time_diff_minutes = len(df_analyse) / 5 / 60

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
    p = Pool()
    for i in range(80):
        p.apply_async(hrv_extraction, args=(i,))
    p.close()
    p.join()


def process_eda_async():
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
