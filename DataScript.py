from multiprocessing import Pool
import pandas as pd

output_folder = 'middle-files/'


def process_time(p_index):
    participant_key = 'P' + str(p_index + 1).zfill(2)
    df = pd.read_csv(f'dataset/{participant_key}/ScreenEvent.csv')
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # find pairs of UNLOCK - OFF
    # essentially each pair means a set time beginning when the user unlocked their phone,
    # and ending when they turned off
    unlocks = df[df['type'] == 'UNLOCK'].index
    pairs = []

    for i in range(len(unlocks)):
        if i < len(unlocks) - 1 and unlocks[i + 1] - unlocks[i] == 1:
            continue
        next_off = df.loc[unlocks[i]:].loc[df['type'] == 'OFF'].head(1)
        if not next_off.empty:
            pairs.append((df.loc[unlocks[i]], next_off.iloc[0]))

    pairs_df = pd.DataFrame({
        'UNLOCK_timestamp': [unlock['timestamp'] for unlock, off in pairs],
        'OFF_timestamp': [off['timestamp'] for unlock, off in pairs]
    })

    pairs_df.to_csv(output_folder + participant_key + '_screen_event_test.csv', index=False)

    # filter and group app usage
    app_df = pd.read_csv(f'dataset/{participant_key}/AppUsageEvent.csv')

    filtered_app_usage = pd.DataFrame()
    screen_pairs = []

    for _, row in pairs_df.iterrows():
        start_time = row['UNLOCK_timestamp']
        end_time = row['OFF_timestamp']
        subset = app_df[(app_df['timestamp'] >= start_time) & (app_df['timestamp'] <= end_time)]

        filtered_app_usage = pd.concat([filtered_app_usage, subset])
        screen_pairs.append({'UNLOCK_timestamp': start_time, 'OFF_timestamp': end_time})

    # consider only certain events
    final_filtered_data = filtered_app_usage[
        (filtered_app_usage['type'] != 'USER_INTERACTION') &
        (filtered_app_usage['category'].isin(['COMMUNICATION', 'SOCIAL'])) &
        (filtered_app_usage['name'].isin(['Facebook', 'Instagram', '카카오톡']))
        ]

    final_filtered_data = final_filtered_data.reset_index(drop=True)
    final_filtered_data.to_csv(output_folder + participant_key + '_test_01.csv', index=False)

    # group events to specific time frame
    time_differences = []
    grouped = final_filtered_data.groupby(['name', 'packageName'])
    for (name, packageName), group in grouped:
        # sort by timestamp
        group = group.sort_values(by='timestamp').reset_index(drop=True)

        # define var to store time specifically for foreground
        foreground_time = None
        for _, row in group.iterrows():
            if row['type'] == 'MOVE_TO_FOREGROUND':
                foreground_time = row['timestamp']

            elif row['type'] == 'MOVE_TO_BACKGROUND' and foreground_time is not None:
                # calculate the difference and store it
                background_time = row['timestamp']
                time_diff = background_time - foreground_time

                # find the associated screen event pair
                for pair in screen_pairs:
                    if pair['UNLOCK_timestamp'] <= foreground_time and pair['OFF_timestamp'] >= background_time:
                        unlock_off_pair = pair
                        break

                time_differences.append({
                    'name': name,
                    'packageName': packageName,
                    'foreground_time': foreground_time,
                    'background_time': background_time,
                    'time_difference': time_diff,
                    'UNLOCK_timestamp': unlock_off_pair['UNLOCK_timestamp'],
                    'OFF_timestamp': unlock_off_pair['OFF_timestamp']
                })

                # reset var
                foreground_time = None

    # create a new df
    time_diff_df = pd.DataFrame(time_differences)
    time_diff_df.to_csv(output_folder + participant_key + '_foreground_background_differences.csv', index=False)

    print(participant_key + ' done')


def process_time_async():
    p = Pool()
    for i in range(80):
        p.apply_async(process_time, args=(i,))
    p.close()
    p.join()

if __name__ == '__main__':
    process_time_async()
