"""
This script is to take the projected co ordinates from the trial data and then compare the PLVs for those to
the PLVs obtained from from the preprocessed data. For this, I need to correctly match the condition 4-6, 5-7 stuff,
use the maps from the raw data to find the corresponding trials in the pre-processed data, and only consider pairs
3-10 inclusive. (1 and 2 have no way of identifying trial starts/ends, 11+ the data is missing).
"""

from common.paths import DATA, MARIUS_CLEAN
from common.data_aggregation import load_preprocessed_movement_data, get_condition_map
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, welch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle




if __name__ == "__main__":
    
    # condition 2 - move alone
    # condition 3 - move together
    # conditions 4, 6 - move and observe
    # conditions 5, 7 - leader follower
    # so all leaders = pp1 from condition 7, pp2 from condition 5
    # so all followers = pp2 from condition 7, pp1 from condition 5
    # so all actors = pp1 from condition 6, pp2 from condition 4
    # so all observers = pp2 from condition 6, pp1 from condition 4
    
    # going to create a dictionary of dictionaries to go: {2 (condition): {pair: [trials for that pair which are that cond]}
    MSFREQ=240 
    condition_names = {'ma' : [2], 'mt' : [3], 'mo' : [4, 6], 'lf' : [5, 7]}

    trial_numbers_new = {2: [i for i in range(1, 17)], 3: [i for i in range(1, 17)], 
                         4: [i for i in range(1, 9)], 5: [i for i in range(1, 9)],
                         6: [i for i in range(1, 9)], 7: [i for i in range(1, 9)]}

    condition_map = {i : {} for i in range(2, 8)}
    print(condition_map)

    easy_pairs = [3, 4, 5, 7, 8, 9, 10]
    for pair in easy_pairs:
        dct = get_condition_map(pair)
        for cond_no in dct:
            if cond_no not in [1, 8]:
                condition_map[cond_no][pair] = dct[cond_no]

    
    print(condition_map)
     
    epoch_starts = [i for i in range(1, 22, 4)]
    for cond in condition_names:
        plvs_preprocessed = []
        plvs_r = []
        for condno in condition_names[cond]:
            for pair in condition_map[condno]: #***
                for trial in condition_map[condno][pair]:
                    p1 = load_preprocessed_movement_data(pair, trial, 1)
                    p2 = load_preprocessed_movement_data(pair, trial, 2)
                    pos1_hilbert = hilbert(p1)
                    pos2_hilbert = hilbert(p2)
                    for e in epoch_starts:
                        p1_phase = np.angle(pos1_hilbert[e*MSFREQ:(e+1)*MSFREQ])
                        p2_phase = np.angle(pos2_hilbert[e*MSFREQ:(e+1)*MSFREQ])
                        phase_diff = p2_phase - p1_phase
                        phase_diff_arr_im = 0*phase_diff+1j*phase_diff
                        plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
                        plvs_preprocessed.append(plv)
                
                for trial in trial_numbers_new[condno]:
                    p1 = pd.read_csv(f'{MARIUS_CLEAN}/projected/condition_{condno}/pair_{pair:02d}/p_{pair:02d}A_trial_{trial:02d}.csv')
                    p2 = pd.read_csv(f'{MARIUS_CLEAN}/projected/condition_{condno}/pair_{pair:02d}/p_{pair:02d}B_trial_{trial:02d}.csv')
                    # print(p1.head())
                    # print(p2.head())
                    r1 = p1['v'].to_list()
                    r2 = p2['v'].to_list()
                    for e in epoch_starts:
                        p1_phase = np.angle(r1[e*MSFREQ:(e+1)*MSFREQ])
                        p2_phase = np.angle(r2[e*MSFREQ:(e+1)*MSFREQ])
                        phase_diff = p2_phase - p1_phase
                        phase_diff_arr_im = 0*phase_diff+1j*phase_diff
                        plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
                        plvs_r.append(plv)
        
        bins = np.linspace(0, 1, 21)          # PLV runs from 0 to 1, so 20 bins of 0.05
        fig, ax = plt.subplots()
        ax.hist(plvs_preprocessed, bins=bins, alpha=0.2, label="preprocessed")
        ax.hist(plvs_r, bins=bins, alpha=0.2, label="projected")
        # ax.hist(plvs_r, bins=bins, alpha=0.2, label="radius")
        ax.set_xlabel("PLV"); ax.set_ylabel("trials")
        ax.legend()
        ax.set_title(f'PLVs for condition {cond}')
        plt.show()

    #epoch_starts = [i for i in range(1, 22, 1)]
    #pos_data = load_preprocessed_movement_data_li(f'{DATA}/marius/processed/pair003_movementdata.mat')
    #for condition in condition_map:
    #    plvs_li = []
    #    good_trials = condition_trial_map[condition]
    #    print(good_trials)
    #    for trial in good_trials:
    #        if trial not in [45, 66]:
    #            pos1_hilbert = hilbert(pos_data[trial][0])
    #            pos2_hilbert = hilbert(pos_data[trial][1])
    #            for e in epoch_starts:
    #                p1_phase = np.angle(pos1_hilbert[e*msfreq:(e+1)*msfreq])
    #                p2_phase = np.angle(pos2_hilbert[e*msfreq:(e+1)*msfreq])
    #                phase_diff = p2_phase - p1_phase
    #                phase_diff_arr_im = 0*phase_diff+1j*phase_diff
    #                plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
    #                plvs_li.append(plv)


    #    plvs_r = []
    #    plvs_projected = []
    #    #get plvs for my data
    #    for trial in good_trials:
    #        if trial not in [45, 66]:
    #            p1 = p1_raw[p1_raw['trial_no'] == trial]['zeta'].to_numpy()
    #            p2 = p2_raw[p2_raw['trial_no'] == trial]['zeta'].to_numpy()
    #            p1_r = p1_raw[p1_raw['trial_no'] == trial]['zeta_r'].to_numpy()
    #            p2_r = p2_raw[p2_raw['trial_no'] == trial]['zeta_r'].to_numpy()
    #            for e in epoch_starts:
    #                p1_phase = np.angle(p1[e*msfreq:(e+1)*msfreq])
    #                p2_phase = np.angle(p2[e*msfreq:(e+1)*msfreq])
    #                p1_phase_r = np.angle(p1_r[e*msfreq:(e+1)*msfreq])
    #                p2_phase_r = np.angle(p2_r[e*msfreq:(e+1)*msfreq])
    #                phase_diff = p2_phase - p1_phase
    #                phase_diff_r = p2_phase_r - p1_phase_r
    #                phase_diff_arr_im = 0*phase_diff+1j*phase_diff
    #                phase_diff_arr_im_r = 0*phase_diff_r+1j*phase_diff_r
    #                plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
    #                plv_r = np.abs(np.mean(np.exp(phase_diff_arr_im_r)))
    #                plvs_projected.append(plv)
    #                plvs_r.append(plv_r)


    # load preporcessed goes pair, trial, participant 
    # p1_t1 = load_preprocessed_movement_data(3, 1, 1)
    # p2_t1 = load_preprocessed_movement_data(3, 1, 2)
    # # plot the x,y,z co-ordinates of one trial data

    # p1_raw_all, p2_raw_all  = load_raw_data(f'{DATA}/marius/raw/polhemus_003_20200129_170637.csv')
    # p1_raw = p1_raw_all[(p1_raw_all['trial_no'] > 0) & (p1_raw_all['trial_no'] != 66) & (p1_raw_all['trial_no'] != 45)]
    # p2_raw = p2_raw_all[(p2_raw_all['trial_no'] > 0) & (p2_raw_all['trial_no'] != 66) & (p2_raw_all['trial_no'] != 45)]
    # xmean, ymean, zmean = np.mean(p1_raw['Position X']), np.mean(p1_raw['Position Y']), np.mean(p1_raw['Position Z'])
    # print(f'the mean of each co-ord is {xmean}, {ymean}, {zmean}. Norm is {np.sqrt(xmean**2 + ymean**2 + zmean**2)}')
    # # fig, axes = plt.subplots(2, 3)
    # # axes[0, 0].scatter(p1_raw['Position X'], p1_raw['Position Y'],s=1)
    # # axes[0, 0].set_xlabel('X')
    # # axes[0, 0].set_ylabel('Y')
    # # axes[0, 1].scatter(p1_raw['Position X'], p1_raw['Position Z'],s=1)
    # # axes[0, 1].set_xlabel('X')
    # # axes[0, 1].set_ylabel('Z')
    # # axes[0, 2].scatter(p1_raw['Position Y'], p1_raw['Position Z'], s=1)
    # # axes[0, 2].set_xlabel('Y')
    # # axes[0, 2].set_ylabel('Z')
    # # plt.show()


    # p1_t1_demeaned = p1_t1 - np.mean(p1_t1)    
    # p1_demeaned_h = hilbert(p1_t1_demeaned)
    # p1_hilbert = hilbert(p1_t1)
    # p2_hilbert = hilbert(p2_t1)
    # # need to handle epochs
    # # first check if the real and imaginary parts go around zero
    
    # # the processed data needs to be demeaned, but it is narrow band.
    # fig, ax = plt.subplots()
    # ax.set_xlabel('real')
    # ax.set_ylabel('im')
    # ax.axvline(0, ls="--", color="k")
    # ax.axhline(0, ls="--", color="k")
    # # ax.plot(np.real(p1_hilbert), np.imag(p1_hilbert), label='normal')
    # ax.plot(np.real(p1_demeaned_h), np.imag(p1_demeaned_h), label='demeaned')
    # plt.show() 

    # f, P = welch(p1_t1, fs=240, nperseg=2048)
    # f0 = f[np.argmax(P)]

    # fig, ax = plt.subplots()
    # ax.semilogy(f, P)
    # ax.set_xlim(0, 3)
    # ax.set_ylim(1e-3, 10)
    # ax.axvline(f0, ls="--", color="k")
    # ax.set_xlabel("frequency (Hz)")
    # ax.set_ylabel("PSD")
    # plt.show()

    # # now let's take the demeaned radius from the raw data, check if it is narrow band or not, then hilbert
    # p1_raw['r_demeaned'] = p1_raw['r'] - np.mean(p1_raw['r'])
    # p2_raw['r_demeaned'] = p2_raw['r'] - np.mean(p2_raw['r'])
    # print(p1_raw.head())
    
    # # f, P = welch(p1_raw['r_demeaned'].to_numpy(), fs=240, nperseg=2048)
    # # f0 = f[np.argmax(P)]

    # # fig, ax = plt.subplots()
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
    # xyz = p1_raw[["Position X", "Position Y", "Position Z"]].to_numpy()
    # p1_raw['projected'] = get_basis_vector(xyz)
    
    # xyz = p2_raw[["Position X", "Position Y", "Position Z"]].to_numpy()
    # p2_raw['projected'] = get_basis_vector(xyz)
    
    # t = np.arange(len(p1_raw))/240
    # r = p1_raw['projected'].to_numpy()

    # p1_raw = p1_raw[p1_raw['trial_no'] >= 1]
    # p1_raw['zeta_r'] = np.nan + 0j                                     # complex column, NaN outside trials
    # p1_raw['zeta'] = np.nan + 0j                                     # complex column, NaN outside trials
    # p1_raw.loc[:,'zeta_r'] = p1_raw.groupby('trial_no')['r_demeaned'].transform(analytic)
    # p1_raw.loc[:,'zeta'] = p1_raw.groupby('trial_no')['projected'].transform(analytic)
    
    # p2_raw = p2_raw[p2_raw['trial_no'] >= 1]
    # p2_raw['zeta_r'] = np.nan + 0j                                     # complex column, NaN outside trials
    # p2_raw['zeta'] = np.nan + 0j                                     # complex column, NaN outside trials
    # p2_raw.loc[:,'zeta_r'] = p2_raw.groupby('trial_no')['r_demeaned'].transform(analytic)
    # p2_raw.loc[:,'zeta'] = p2_raw.groupby('trial_no')['projected'].transform(analytic)


    # p1_raw['t'] = p1_raw.groupby('trial_no').cumcount()
    # p2_raw['t'] = p2_raw.groupby('trial_no').cumcount()

    
    # p1_raw['t']

    #fig, ax = plt.subplots()
    #ax.set_xlabel('real')                                                    
    #ax.set_ylabel('im')                                                      
    #ax.axvline(0, ls="--", color="k")                                        
    #ax.axhline(0, ls="--", color="k")                                        
    #for _, g in p2_raw.groupby('trial_no'):
    #    z = g['zeta'].to_numpy()[480:-480]        # drop 2 s at each end
    #    ax.plot(np.real(z), np.imag(z), lw=0.5) 
    #ax.set_title('Real and imaginary parts of hilbert transform applied to the projected vector')
    #plt.show()      
    #e = 1
    #msfreq = 240
    
    #with open(f'{DATA}/marius/processed/pair003_conditions.pkl', 'rb') as f:
    #    condition_data = pickle.load(f)[0]

    #print(condition_data)
    #print(f'the condition data object is of type {type(condition_data)}, each element is of type {type(condition_data[0])}, each element of that is of type {type(condition_data[0][0])}')
    #condition_trial_map = {}
    #for i in range(1, 9):
    #    condition_trial_map[i] = []

    #for pair in condition_data:
    #    condition_trial_map[int(pair[1])].append(int(pair[0]) - 1)
    


    ##epoch_starts = [i for i in range(1, 22, 1)]
    #pos_data = load_preprocessed_movement_data_li(f'{DATA}/marius/processed/pair003_movementdata.mat')
    #for condition in condition_trial_map:
    #    plvs_li = []
    #    good_trials = condition_trial_map[condition]
    #    print(good_trials)
    #    for trial in good_trials:
    #        if trial not in [45, 66]:
    #            pos1_hilbert = hilbert(pos_data[trial][0])
    #            pos2_hilbert = hilbert(pos_data[trial][1])
    #            for e in epoch_starts:
    #                p1_phase = np.angle(pos1_hilbert[e*msfreq:(e+1)*msfreq])
    #                p2_phase = np.angle(pos2_hilbert[e*msfreq:(e+1)*msfreq])
    #                phase_diff = p2_phase - p1_phase
    #                phase_diff_arr_im = 0*phase_diff+1j*phase_diff
    #                plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
    #                plvs_li.append(plv)


    #    plvs_r = []
    #    plvs_projected = []
    #    #get plvs for my data
    #    for trial in good_trials:
    #        if trial not in [45, 66]:
    #            p1 = p1_raw[p1_raw['trial_no'] == trial]['zeta'].to_numpy()
    #            p2 = p2_raw[p2_raw['trial_no'] == trial]['zeta'].to_numpy()
    #            p1_r = p1_raw[p1_raw['trial_no'] == trial]['zeta_r'].to_numpy()
    #            p2_r = p2_raw[p2_raw['trial_no'] == trial]['zeta_r'].to_numpy()
    #            for e in epoch_starts:
    #                p1_phase = np.angle(p1[e*msfreq:(e+1)*msfreq])
    #                p2_phase = np.angle(p2[e*msfreq:(e+1)*msfreq])
    #                p1_phase_r = np.angle(p1_r[e*msfreq:(e+1)*msfreq])
    #                p2_phase_r = np.angle(p2_r[e*msfreq:(e+1)*msfreq])
    #                phase_diff = p2_phase - p1_phase
    #                phase_diff_r = p2_phase_r - p1_phase_r
    #                phase_diff_arr_im = 0*phase_diff+1j*phase_diff
    #                phase_diff_arr_im_r = 0*phase_diff_r+1j*phase_diff_r
    #                plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
    #                plv_r = np.abs(np.mean(np.exp(phase_diff_arr_im_r)))
    #                plvs_projected.append(plv)
    #                plvs_r.append(plv_r)


    #    bins = np.linspace(0, 1, 21)          # PLV runs from 0 to 1, so 20 bins of 0.05
    #    fig, ax = plt.subplots()
    #    ax.hist(plvs_li, bins=bins, alpha=0.2, label="Li's")
    #    ax.hist(plvs_projected, bins=bins, alpha=0.2, label="projected")
    #    # ax.hist(plvs_r, bins=bins, alpha=0.2, label="radius")
    #    ax.set_xlabel("PLV"); ax.set_ylabel("trials")
    #    ax.legend()
    #    ax.set_title(f'PLVs for condition {condition}')
    #    plt.show()

    #    # pos1_phase = np.angle(pos1_hilbert[e*msfreq:(e+1)*msfreq])
    #    # pos2_phase = np.angle(pos2_hilbert[e*msfreq:(e+1)*msfreq])
    #    # p2_p1_phase_diff = pos2_phase - pos1_phase
    ## # Convert phase difference to complex part i(phase1-phase2)
    ## phase_diff_arr_im = 0*p2_p1_phase_diff+1j*p2_p1_phase_diff
    ## # Perform exp, mean and abs
    ## plv = np.abs(np.mean(np.exp(phase_diff_arr_im)))
    # print(f'plv is {plv}')


    





