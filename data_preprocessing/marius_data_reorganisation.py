from common.paths import DATA, MARIUS_RAW, MARIUS_CLEAN
from pathlib import Path
import numpy as np
import pandas as pd

LEN_T1 = 28801
T_LEN = 6241 
"""
Original Marius' data files were quite messy/confusing. This is an attempt to make it more clear. The original files look like:

 Sensor|Sample|Timestamp|Marker|Position X|Position Y|Position Z|Orientation X|Orientation Y|Orientation Z
   1   |  51  |   204   |   1  |+013.111  |-007.14   |-004.181  |+082.024     |-021.93      |-096.81
   2   |  51  |   204   |   1  |+012.293  |+003.69   |-002.235  |+178.843     |-049.76      |+102.78

Sensor 1, sensor 2 essentially is participant 1, 2. CSVs alternate line by line one entry from participant 1, one from 
participant 2. Hence Timesteamp is the same for 2 rows. Marker can be used (pattern found with an AI agent) to find the 
start of a trial. 1 120s trial at the start (which is condition 1), one at the end (which is condition 8) and then 
80 trials in the middle lasting 25s (according to the logfile_trial_order file), but they seem to vary a bit in practice.
Trials are separated by 10s gaps. Each trial is one of 6 conditions (technically 4, original layout very confusing, explained in 
readme file), order of conditions varies per pair, logfile_trial_order file has the map of trial_no:condition. 
Position columns are xyz co-ords, orientation columns not used. Unsure whatsample is. 

All pair trials accross all conditions are in one large csv. This script takes the original large csv and does the
following:
    1. splits the csv into participant 1 and 2 (which are later labeled A and B, respectively, in filenames).
       Calculates the radius (sqrt(x^2 + y^2 + z^2))
    2. uses the pattern in marker (6 values of 0 spaces 1s (240 rows) apart, then 0.1 seconds later another 0)
       to split the rows into trials, gaps in between trials are discarded. Populates 'trial_no' column with this info.
    3. uses the trial order log file to populate the 'condition' column.
       As an example, for pair 3, the following trials are from condition 2 (first trial = 1): 
       [3, 7, 14, 19, 24, 27, 34, 39, 46, 49, 52, 57, 64, 67, 76, 80]
    4. creates a column 'new_trial_no' which maps these trial numbers to 1-n, to make it easier. New trial numbers 
       remain consistent across candidates within a pair, e.g. pair003 candidate A new trial 1 is paired with 
       pair003 candidate B new trial 1
    5. creates a new timestamp column 't', which is just 1-n, to get real time need to divide by 240 (recording freq).
    6. Keeps all columns from the original dataframe, but renames Pos. X, Y, Z to x, y, z for ease later.
    7. saves the file in the same format as I have saved Kyveli's data: condition_{i}/pair_00{j}/p_01A_trial_0{k}.csv
       which corresponds to condition i, pair j, participant A and trial k

After this, the co-ordinate problem needs to be addressed, and the files corresponding to the same pair of people in 
different roles for a condition need to be matched up. (This sentence is somewhat confusing, see README file for an 
(attempted) explanation on Marius' naming conventions).

The csvs created by this script look like:
Sensor|Sample|Timestamp|Marker|  x  |  y  |  z  |Orientation X|Orientation Y|Orientation Z|  r  |trial_no|condition|new_trial_no|t
   1  |  3733|    15545|     0|13.21|-7.16|-4.21|80.761299    |-21.589529   |-99.631981   |15.61|    0   |    1    |     1      |0
   1  |  3734|    15549|     1|13.21|-7.17|-4.20|80.760414    |-21.594881   |-99.647133   |15.61|    0   |    1    |     1      |1



"""

def get_trials_conditions(raw_data, pair):
    """
    Take the raw dataframe that corresponds to a participant. Use the marker column to identify the trial start points 
    and the lengths above to find the ends. Add a column to show what trial it is.

    if the gap is roughly 0.1 second and the gap before is 
    roughly 1 second. (the lower bound of 0.05 is needed because)
    there are usually two zeroes in a row. The second check removes
    the pulses that occur mid trial.
    """
    # start with raw dataframe for just one participant
    marker = raw_data['Marker'].to_numpy()  
    marker_shifted = np.concat([[1], marker[:-1]])
    # The points that identify a pulse are where there is a 1 followed by a zero in the marker.
    pulses = np.where((marker == 0) & (marker_shifted == 1))[0]
    gaps = np.diff(pulses)/240 # convert the gap between pulses into seconds. Trial starts after two pulses that 
                               # are 0.1s apart, which come after 5 pulses 1s apart
    starts = []
    for i in range(2, len(pulses)):
        if 0.05 < gaps[i-1] < 0.15 and 0.8 < gaps[i-2] < 1.2: 
            starts.append(pulses[i])
    
    raw_data['trial_no'] = -1
    raw_data['condition'] = -1
    trial_col = raw_data.columns.get_loc('trial_no')
    cond_col = raw_data.columns.get_loc('condition')
    
    condition_map_path = list(MARIUS_RAW.glob(f'pair{pair:03d}/logfile_trialorder*'))[0]
    condition_trial_map = {}
    with open(condition_map_path) as file:
        for idx, line in enumerate(file):
            condition_trial_map[idx] = int(line[0])
    
    for idx, start in enumerate(starts):
        if idx == 0 or idx == len(starts) -1: #long trial at start and end
            n = LEN_T1
        else:
            n = T_LEN
        
        raw_data.iloc[start:start+n, trial_col] = idx # assign a trial number to the n rows starting at the start val
        raw_data.iloc[start:start+n, cond_col] = condition_trial_map[idx] # assign a condition number to the n rows starting at the start val
        
    return raw_data

def load_raw_data(pair):
    """
    The raw data has alternating rows for participant 1 and participant 2. All trials are together for each pair in a
    single csv file. The marker column has some identifiable pattern that marks the beginning of trials.
    Using AI to find this pattern to divide the raw data into trials.
    paths = sorted(DATA.glob('condition_*/pair_*/p_*_trial_*.csv'))

    """
    data_path = list(MARIUS_RAW.glob(f'pair{pair:03d}/*.csv'))[0]
    raw = pd.read_csv(data_path)
    raw = raw.rename(columns={'Position X': 'x', 'Position Y': 'y', 'Position Z': 'z'})
    raw['r'] = np.sqrt(raw['x']**2 + raw['y']**2 + raw['z']**2)
    raw_p1 = raw[raw['Sensor'] == 1]
    raw_p2 = raw[raw['Sensor'] == 2]
    
    pp1 = get_trials_conditions(raw_p1, pair)
    pp2 = get_trials_conditions(raw_p2, pair)
    
    return pp1, pp2


def assign_new_columns_and_save(df, pair, participant):
    participant_label = "A" if participant == 1 else "B"
    cond_vals = df['condition'].unique()
    for condition_no in cond_vals:
        if condition_no == -1:
            continue
        Path(f"{MARIUS_CLEAN}/condition_{condition_no}/pair_{pair:02d}").mkdir(parents=True, exist_ok=True)
        outdir = f"{MARIUS_CLEAN}/condition_{condition_no}/pair_{pair:02d}"
        print(outdir)
        condition_df = pp1[pp1['condition'] == condition_no]
        trial_values = condition_df['trial_no'].unique()
        df_map = pd.DataFrame([[trial_values[i], i + 1] for i in range(len(trial_values))], 
                                  columns = ["old_trial_no", "new_trial_no"])
        condition_df['new_trial_no'] = condition_df['trial_no'].map(df_map.set_index('old_trial_no').squeeze())
        trial_file_labels = condition_df['new_trial_no'].unique()
        for trial_no in trial_file_labels:
            trial_df = condition_df[condition_df['new_trial_no'] == trial_no]
            trial_df['t'] = np.arange(len(trial_df))
            trial_df.to_csv(f"{outdir}/p_{pair:02d}{participant_label}_trial_{trial_no:02d}.csv", index=False)


if __name__ == "__main__":
    pair = 3
    pp1, pp2 = load_raw_data(pair)
    assign_new_columns_and_save(pp1, 3, 1)
    
