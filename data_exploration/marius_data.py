from common.paths import DATA
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, welch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

LEN_T1 = 28801
T_LEN = 6241 
MYSTERY_CONSTANTS = {1: 15.6752, 2: 14.4021} # from the raw data radius, these values were subtracted from the smoothed
                                             # radius to obtain the values they have
def load_preprocessed_movement_data_li(file):
    """
    Li's func for getting the processed data. Pass file name and it handles the indexing of the
    .mat files. Files are all trials for one pair. Returns list of lists, each element of sublist
    is array corresponding to participants position.
    
    first trial is some kind of calibration, 28801 entries @ 240Hz = 120s of recording (plus one element (?))
    succeeding trials are 6241 entries @ 240Hz = 26 seconds of recording (plus one element).

    """
    tmp_dict = loadmat(file)
    tmp_arr = tmp_dict["polhemus"] # get the polhemus data
    # Each entry in tmp_arr correspond to a trial
    # For each trial, there is the raw positions, but also computed features
    # We will start by retrieving the raw positions
    n_trials = len(tmp_arr)
    # print(f"Number of trials: {n_trials}")
    pos_data = [0]*n_trials
    for t in range(n_trials):
        ppn1_pos = tmp_arr[t][0][0][0][0][0][0][0].ravel()
        ppn2_pos = tmp_arr[t][0][0][0][0][0][0][1].ravel()
        pos_data[t] = [ppn1_pos, ppn2_pos]

    return pos_data

def load_preprocessed_movement_data(pair, trial, participant):
    mat = loadmat(f'{DATA}/marius/processed/pair{pair:03d}_movementdata.mat',
                squeeze_me=True, struct_as_record=False)

    tr = mat['polhemus'][trial] 
    if participant == 1:
        ppn = tr.pos.ppn1
    elif participant == 2:
        ppn = tr.pos.ppn2
    else:
        ppn = [tr.pos.ppn1, tr.pos.ppn2]
    return ppn


def get_trials(raw_data):
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
    col = raw_data.columns.get_loc('trial_no')
    
    for idx, start in enumerate(starts):
        if idx == 0 or idx == len(starts) -1: #long trial at start and end
            n = LEN_T1
            raw_data.iloc[start:start+n, col] = 0 # assign a trial number to the n rows starting at the start val
        else:
            n = T_LEN
            raw_data.iloc[start:start+n, col] = idx # assign a trial number to the n rows starting at the start val

    return raw_data

def load_raw_data(file):
    """
    The raw data has alternating rows for participant 1 and participant 2. All trials are together for each pair in a
    single csv file. The marker column has some identifiable pattern that marks the beginning of trials.
    Using AI to find this pattern to divide the raw data into trials.
    """
    raw = pd.read_csv(file)
    raw['r'] = np.sqrt(raw['Position X']**2 + raw['Position Y']**2 + raw['Position Z']**2)
    raw_p1 = raw[raw['Sensor'] == 1]
    raw_p2 = raw[raw['Sensor'] == 2]
    
    pp1 = get_trials(raw_p1)
    pp2 = get_trials(raw_p2)
    
    return pp1, pp2

def raw_to_preprocessed(df):
    participant = df['Sensor'].to_numpy()[0]
    df['r_smooth'] = uniform_filter1d(df['r'].to_numpy(), 24, mode="nearest")
    df['r_smooth'] = df['r_smooth'] - MYSTERY_CONSTANTS[participant]
    return df

def get_basis_vector(xyz):
    centre = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - centre, full_matrices=False)
    direction = vt[0] # unit vector along the track
    s = (xyz - centre) @ direction
    return s


def analytic(s):
    x = s.to_numpy()
    return hilbert(x - x.mean())


if __name__ == "__main__":
    # I now have 3 ways of loading the same data. Li's fn for loading the preprocessed data, mine for doing the same 
    # thing, and a fn that loads the raw data and then applies the processing that Claude found gets me from raw
    # to processed.

    # Want to check that all 3 are the same.
    pos_data = load_preprocessed_movement_data_li(f'{DATA}/marius/processed/pair003_movementdata.mat')
    t1_li = pos_data[12][1] #first trial for first participant
    t1_me = load_preprocessed_movement_data(3, 12, 2)
    p1, p2 = load_raw_data(f'{DATA}/marius/raw/polhemus_003_20200129_170637.csv')
    p1 = raw_to_preprocessed(p1)
    p2 = raw_to_preprocessed(p2)
    print(p1.head())
    print(f'Lis data has len {len(t1_li)}, mine has {len(t1_me)}, raw to processed has {len(p1[p1['trial_no'] == 1])}')    
    p1_t1 = p2[p2['trial_no'] == 12]['r_smooth'].to_numpy()
    
    t = np.arange(len(t1_li))
    fig, ax = plt.subplots()
    ax.plot(t, t1_me, linestyle='--', label = 'My pre processed data')
    ax.plot(t, p1_t1, linestyle='-.', label = 'data from raw')
    ax.plot(t, t1_li, linestyle=':', label = 'Li\'s data')
    plt.legend()
    plt.show()
    
    # All 3 give the same, perfecto
    # next is Li's code applying the hilbert transform to the data
    
    p1_t1 = load_preprocessed_movement_data(3, 1, 1)
    p2_t1 = load_preprocessed_movement_data(3, 1, 2)
    # plot the x,y,z co-ordinates of one trial data

    p1_raw_all, p2_raw_all  = load_raw_data(f'{DATA}/marius/raw/polhemus_003_20200129_170637.csv')
    p1_raw = p1_raw_all[(p1_raw_all['trial_no'] > 0) & (p1_raw_all['trial_no'] != 66) & (p1_raw_all['trial_no'] != 45)]
    p2_raw = p2_raw_all[(p2_raw_all['trial_no'] > 0) & (p2_raw_all['trial_no'] != 66) & (p2_raw_all['trial_no'] != 45)]
    xmean, ymean, zmean = np.mean(p1_raw['Position X']), np.mean(p1_raw['Position Y']), np.mean(p1_raw['Position Z'])
    print(f'the mean of each co-ord is {xmean}, {ymean}, {zmean}. Norm is {np.sqrt(xmean**2 + ymean**2 + zmean**2)}')
    # fig, axes = plt.subplots(2, 3)
    # axes[0, 0].scatter(p1_raw['Position X'], p1_raw['Position Y'],s=1)
    # axes[0, 0].set_xlabel('X')
    # axes[0, 0].set_ylabel('Y')
    # axes[0, 1].scatter(p1_raw['Position X'], p1_raw['Position Z'],s=1)
    # axes[0, 1].set_xlabel('X')
    # axes[0, 1].set_ylabel('Z')
    # axes[0, 2].scatter(p1_raw['Position Y'], p1_raw['Position Z'], s=1)
    # axes[0, 2].set_xlabel('Y')
    # axes[0, 2].set_ylabel('Z')
    # plt.show()


    p1_t1_demeaned = p1_t1 - np.mean(p1_t1)    
    p1_demeaned_h = hilbert(p1_t1_demeaned)
    p1_hilbert = hilbert(p1_t1)
    p2_hilbert = hilbert(p2_t1)
    # need to handle epochs
    # first check if the real and imaginary parts go around zero
    
    # the processed data needs to be demeaned, but it is narrow band.
    fig, ax = plt.subplots()
    ax.set_xlabel('real')
    ax.set_ylabel('im')
    ax.axvline(0, ls="--", color="k")
    ax.axhline(0, ls="--", color="k")
    # ax.plot(np.real(p1_hilbert), np.imag(p1_hilbert), label='normal')
    ax.plot(np.real(p1_demeaned_h), np.imag(p1_demeaned_h), label='demeaned')
    plt.show() 

    f, P = welch(p1_t1, fs=240, nperseg=2048)
    f0 = f[np.argmax(P)]

    fig, ax = plt.subplots()
    ax.semilogy(f, P)
    ax.set_xlim(0, 3)
    ax.set_ylim(1e-3, 10)
    ax.axvline(f0, ls="--", color="k")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    plt.show()

    # now let's take the demeaned radius from the raw data, check if it is narrow band or not, then hilbert
    p1_raw['r_demeaned'] = p1_raw['r'] - np.mean(p1_raw['r'])
    p2_raw['r_demeaned'] = p2_raw['r'] - np.mean(p2_raw['r'])
    print(p1_raw.head())
    
    # f, P = welch(p1_raw['r_demeaned'].to_numpy(), fs=240, nperseg=2048)
    # f0 = f[np.argmax(P)]

    # fig, ax = plt.subplots()
    # ax.semilogy(f, P)
    # ax.set_xlim(0, 3)
    # ax.set_ylim(1e-3, 10)
    # ax.axvline(f0, ls="--", color="k")
    # ax.set_xlabel("frequency (Hz)")
    # ax.set_ylabel("PSD")
    # ax.set_title('power spectrum of the demeaned radius')
    # plt.show()
    
    # p1_hilbert = hilbert(p1_raw['r_demeaned'].to_numpy())
    # fig, ax = plt.subplots()
    # ax.set_xlabel('real')                                                    
    # ax.set_ylabel('im')                                                      
    # ax.axvline(0, ls="--", color="k")                                        
    # ax.axhline(0, ls="--", color="k")                                        
    # # ax.plot(np.real(p1_hilbert), np.imag(p1_hilbert), label='normal')      
    # ax.plot(np.real(p1_hilbert), np.imag(p1_hilbert))
    # ax.set_title('Real and imaginary parts of hilbert transform applied to demeaned rad')
    # plt.show()      
    
    # # Okay, so the above signal works basically, but let's see how it looks if I project onto the vector
    xyz = p1_raw[["Position X", "Position Y", "Position Z"]].to_numpy()
    p1_raw['projected'] = get_basis_vector(xyz)
    
    xyz = p2_raw[["Position X", "Position Y", "Position Z"]].to_numpy()
    p2_raw['projected'] = get_basis_vector(xyz)
    
    t = np.arange(len(p1_raw))/240
    r = p1_raw['projected'].to_numpy()

    p1_raw = p1_raw[p1_raw['trial_no'] >= 1]
    p1_raw['zeta_r'] = np.nan + 0j                                     # complex column, NaN outside trials
    p1_raw['zeta'] = np.nan + 0j                                     # complex column, NaN outside trials
    p1_raw.loc[:,'zeta_r'] = p1_raw.groupby('trial_no')['r_demeaned'].transform(analytic)
    p1_raw.loc[:,'zeta'] = p1_raw.groupby('trial_no')['projected'].transform(analytic)
    
    p2_raw = p2_raw[p2_raw['trial_no'] >= 1]
    p2_raw['zeta_r'] = np.nan + 0j                                     # complex column, NaN outside trials
    p2_raw['zeta'] = np.nan + 0j                                     # complex column, NaN outside trials
    p2_raw.loc[:,'zeta_r'] = p2_raw.groupby('trial_no')['r_demeaned'].transform(analytic)
    p2_raw.loc[:,'zeta'] = p2_raw.groupby('trial_no')['projected'].transform(analytic)


    # p1_raw['t'] = p1_raw.groupby('trial_no').cumcount()
    # p2_raw['t'] = p2_raw.groupby('trial_no').cumcount()

    
    # p1_raw['t']

    fig, ax = plt.subplots()
    ax.set_xlabel('real')                                                    
    ax.set_ylabel('im')                                                      
    ax.axvline(0, ls="--", color="k")                                        
    ax.axhline(0, ls="--", color="k")                                        
    for _, g in p2_raw.groupby('trial_no'):
        z = g['zeta'].to_numpy()[480:-480]        # drop 2 s at each end
        ax.plot(np.real(z), np.imag(z), lw=0.5) 
    ax.set_title('Real and imaginary parts of hilbert transform applied to the projected vector')
    plt.show()      
    e = 1
    msfreq = 240
    
    with open(f'{DATA}/marius/processed/pair003_conditions.pkl', 'rb') as f:
        condition_data = pickle.load(f)[0]

    print(condition_data)
    print(f'the condition data object is of type {type(condition_data)}, each element is of type {type(condition_data[0])}, each element of that is of type {type(condition_data[0][0])}')
    condition_trial_map = {}
    for i in range(1, 9):
        condition_trial_map[i] = []

    for pair in condition_data:
        condition_trial_map[int(pair[1])].append(int(pair[0]) - 1)
    


    epoch_starts = [i for i in range(1, 22, 1)]
    pos_data = load_preprocessed_movement_data_li(f'{DATA}/marius/processed/pair003_movementdata.mat')
    for condition in condition_trial_map:
        plvs_li = []
        good_trials = condition_trial_map[condition]
        print(good_trials)
        for trial in good_trials:
            if trial not in [45, 66]:
                pos1_hilbert = hilbert(pos_data[trial][0])
                pos2_hilbert = hilbert(pos_data[trial][1])
                for e in epoch_starts:
                    p1_phase = np.angle(pos1_hilbert[e*msfreq:(e+1)*msfreq])
                    p2_phase = np.angle(pos2_hilbert[e*msfreq:(e+1)*msfreq])
                    phase_diff = p2_phase - p1_phase
                    phase_diff_arr_im = 0*phase_diff+1j*phase_diff
                    plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
                    plvs_li.append(plv)


        plvs_r = []
        plvs_projected = []
        #get plvs for my data
        for trial in good_trials:
            if trial not in [45, 66]:
                p1 = p1_raw[p1_raw['trial_no'] == trial]['zeta'].to_numpy()
                p2 = p2_raw[p2_raw['trial_no'] == trial]['zeta'].to_numpy()
                p1_r = p1_raw[p1_raw['trial_no'] == trial]['zeta_r'].to_numpy()
                p2_r = p2_raw[p2_raw['trial_no'] == trial]['zeta_r'].to_numpy()
                for e in epoch_starts:
                    p1_phase = np.angle(p1[e*msfreq:(e+1)*msfreq])
                    p2_phase = np.angle(p2[e*msfreq:(e+1)*msfreq])
                    p1_phase_r = np.angle(p1_r[e*msfreq:(e+1)*msfreq])
                    p2_phase_r = np.angle(p2_r[e*msfreq:(e+1)*msfreq])
                    phase_diff = p2_phase - p1_phase
                    phase_diff_r = p2_phase_r - p1_phase_r
                    phase_diff_arr_im = 0*phase_diff+1j*phase_diff
                    phase_diff_arr_im_r = 0*phase_diff_r+1j*phase_diff_r
                    plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
                    plv_r = np.abs(np.mean(np.exp(phase_diff_arr_im_r)))
                    plvs_projected.append(plv)
                    plvs_r.append(plv_r)


        bins = np.linspace(0, 1, 21)          # PLV runs from 0 to 1, so 20 bins of 0.05
        fig, ax = plt.subplots()
        ax.hist(plvs_li, bins=bins, alpha=0.2, label="Li's")
        ax.hist(plvs_projected, bins=bins, alpha=0.2, label="projected")
        # ax.hist(plvs_r, bins=bins, alpha=0.2, label="radius")
        ax.set_xlabel("PLV"); ax.set_ylabel("trials")
        ax.legend()
        ax.set_title(f'PLVs for condition {condition}')
        plt.show()

        # pos1_phase = np.angle(pos1_hilbert[e*msfreq:(e+1)*msfreq])
        # pos2_phase = np.angle(pos2_hilbert[e*msfreq:(e+1)*msfreq])
        # p2_p1_phase_diff = pos2_phase - pos1_phase
    # # Convert phase difference to complex part i(phase1-phase2)
    # phase_diff_arr_im = 0*p2_p1_phase_diff+1j*p2_p1_phase_diff
    # # Perform exp, mean and abs
    # plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
    # print(f'plv is {plv}')


    





