from multiprocessing import Pool
import pandas as pd

output_folder = 'middle-files/'

# function to calculate average confidence levels
def calculate_average_confidence(start_time, end_time, activity_df):
    relevant_activities = activity_df[(activity_df['timestamp'] >= start_time) & (activity_df['timestamp'] <= end_time)]
    if not relevant_activities.empty:
        average_confidences = relevant_activities.mean(numeric_only=True).round(4)
        # print(sum(average_confidences.to_dict().values()))
        # confidence_sum = sum(current_row[col] for col in confidence_columns)
        # result = average_confidences.to_dict()
        # sum = 0
        # for key in result:
        #     if key != 'timestamp':
        #         print('key: ', key, '-', result[key])
        #         sum += result[key]
        # print('sum: ', sum)
        return average_confidences.to_dict()
    else:
        return {col: 0.0 for col in activity_df.columns if col != 'timestamp'}

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
    # final_filtered_data.to_csv(output_folder + participant_key + '_test_01.csv', index=False)

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
    time_diff_df = time_diff_df.sort_values(by='foreground_time').reset_index(drop=True)

    result = []

    activity_df = pd.read_csv(f'dataset/{participant_key}/ActivityEvent.csv')

    # Calculate initial average confidence levels for each app usage event
    confidence_columns = ['confidenceStill', 'confidenceUnknown', 'confidenceOnFoot', 'confidenceWalking', 'confidenceInVehicle', 'confidenceOnBicycle', 'confidenceRunning', 'confidenceTilting']
    for col in confidence_columns:
        time_diff_df[col] = 0.0

    for index, row in time_diff_df.iterrows():
        # modify the below line to only consider a set time before and after foreground time
        average_confidences = calculate_average_confidence(row['foreground_time']-120000, row['foreground_time']+12000, activity_df)
        for col in confidence_columns:
            time_diff_df.at[index, col] = round(average_confidences.get(col, 0.0), 4)
    # iterate over the app usage events dataframe to combine app usage event and find appropriate activity time
    i = 0
    while i < len(time_diff_df):
        # get current row details
        current_row = time_diff_df.iloc[i]
        current_app = current_row['packageName']
        unlock_time = current_row['UNLOCK_timestamp']
        off_time = current_row['OFF_timestamp']
        start_time = current_row['foreground_time']
        end_time = current_row['background_time']

        # check for sum of confidences; skip current row if == 0.0
        confidence_sum = sum(current_row[col] for col in confidence_columns)
        #print('current row ', i, ' - ', confidence_sum)
        if confidence_sum == 0.0:
            #print('skip row ', i)
            i += 1
            continue
        
        # the idea is to get the average confidence level based on the existing confidences
        # create a dictionary to store the sums of each type of confidences and the count
        cumulative_confidences = {col: current_row[col] for col in confidence_columns}
        count = 1 
        
        # check consecutive rows for the same app and within the same UNLOCK-OFF window
        j = i + 1
        while j < len(time_diff_df):
            next_row = time_diff_df.iloc[j]
            
            # break the loop in case of different app or different time window
            if (next_row['packageName'] != current_app or 
                next_row['UNLOCK_timestamp'] != unlock_time or 
                next_row['OFF_timestamp'] != off_time):
                break

            # check for sum of confidences; skip current row if == 0.0
            confidence_sum = sum(next_row[col] for col in confidence_columns)
            #print('current row ', j, ' - ', confidence_sum)
            if confidence_sum == 0.0:
                #print('skip row ', j)
                j += 1
                continue
            
            # the max time difference is set to 30000 milliseconds (30 seconds) to combine app events
            if next_row['foreground_time'] - end_time < 30000:
                end_time = next_row['background_time']
                for col in confidence_columns:
                    cumulative_confidences[col] += next_row[col]
                count += 1
                j += 1
            else:
                break
        
        average_confidences_combined = {col: round(cumulative_confidences[col] / count, 4) for col in confidence_columns}
        combined_event = {
            'name': current_row['name'],
            'packageName': current_app,
            'foreground_time': start_time,
            'background_time': end_time,
            'time_difference': end_time - start_time,
            'UNLOCK_timestamp': unlock_time,
            'OFF_timestamp': off_time
        }
        combined_event.update(average_confidences_combined)
        result.append(combined_event)
        # temp = sum(current_row[col] for col in confidence_columns)
        # print('current row ', i, ' - ', temp)
        i = j

    result_df = pd.DataFrame(result)
    result_df.to_csv(output_folder + participant_key + '_app_usage_events_with_confidences.csv', index=False)

    print(participant_key + ' done')

def process_time_async():
    p = Pool()
    for i in range(80):
        p.apply_async(process_time, args=(i,))
    p.close()
    p.join()

if __name__ == '__main__':
    process_time_async()