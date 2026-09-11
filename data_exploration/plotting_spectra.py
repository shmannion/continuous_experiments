from common.paths import C1, C2, C5, C6
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import hilbert, welch, spectrogram

"""
I have 6 conditions in this experiment. 
naming convention: 
    {cond}_t{trial_no}_p{participant_no}
    solo:  cond 1 - solo (don't see one another, told "make interesting movements")
    spont: cond 2 - spontaneous sync (told same as 1, can see each other)
    self:  cond 3 - focus on yourself. Each player focuses on their own movements.
    other: cond 4 - focus on the other player's movements
    p1:    cond 5 - both people focus on player 1.
    p2:    cond 6 - both people focus on player 2.


Following guide in Pikovsky, Rosenblum & Kurths Appendix A2
scipy.signal.hilbert returns zeta(t) = s(t) + i s_H(t) = A(t)e^{i \ phi(t)}

"""

def calculate_phase(df: pd.DataFrame):
    df.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 
    df['zeta'] = hilbert(df['r'])
    df['zeta_demeaned'] = hilbert(df['r'] - df['r'].mean())
    
    #N.B. This next line is incorrect as of 3/9/26 @ 16:50, unwrap needs to take the band-passed version
    # df['theta'] = np.unwrap(np.angle(df['H']))
    return df

def get_r(path):
    df = pd.read_csv(path, header=None)
    df = df.iloc[120:]
    df.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 
    return df['z'].to_numpy()

def get_paths(root):
    """
    Get all of the paths for condition 1-6
    """
    paths = sorted(root.glob('pair_*/p_*_trial_*.csv'))
    return paths

def get_condition_r(root, candidate):
    """
    Gets just the norm of the xyz co-ordinates for the condition (passed as root, C1-6)
    Takes candidate as one of 'A', 'B', 'AB' 
    """
    rows = []
    paths = get_paths(root)
    for path in paths:
        m = re.match(rf"p_(\d+)([AB])_trial_(\d+)", path.stem)
        pair = int(m[1])
        participant = m[2]
        trial = int(m[3])
        if participant not in candidate:
            continue
        r = get_r(path)
        # f, P = welch(r - r.mean(), fs=240, nperseg=2048)
        res = {'pair': pair, 'participant': participant, 'trial': trial, 'r' : r}
        rows.append(res)

    return rows

def get_power_spectra(rows):
    spectra = []
    for row in rows:
        f, P = welch(row['r'] - np.mean(row['r']), fs=240, nperseg=4800)
        row['f0'] = f[np.argmax(P)]
        spectra.append(P)

    return f, rows, np.array(spectra)


if __name__ == "__main__":
    df1 = pd.read_csv(f'{C5}/pair_01/p_01A_trial_01.csv', header=None)

    rows = get_condition_r(C5, 'AB')
    print(rows)
    f, rows, spectra = get_power_spectra(rows)
    peaks = pd.DataFrame(rows)
    peaks["f0"].hist(bins=20)
    plt.show()
    # f = np.arange(0, 120, 240/2048)
    fig, ax = plt.subplots()
    ax.semilogy(f, spectra.T, color="grey", alpha=0.2)   # every trial faintly
    ax.semilogy(f, spectra.mean(axis=0), color="k")     # the average
    ax.semilogy(f, np.median(spectra, axis=0), color="red")     # the median
    ax.set_xlim(0, 3)
    ax.set_ylim(1e-3, 100)
    plt.show()

    print(peaks.groupby(["pair", "participant"])["f0"].describe())

    peaks = peaks.sort_values("f0")
    slow = peaks.head(6)                            # lowest f0
    fast = peaks[peaks["f0"].between(0.7, 1.0)].head(6)

    fig, axes = plt.subplots(6, 2, figsize=(12, 12), sharex=True)
    for col, group in enumerate([slow, fast]):
        for ax, (_, row) in zip(axes[:, col], group.iterrows()):
            r = row["r"]
            t = np.arange(len(r)) / 240
            ax.plot(t, r - r.mean())
            ax.set_title(f"pair {row.pair}{row.participant} trial {row.trial}, f0={row.f0:.2f} Hz", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)")
    plt.show()
    # f, P = welch(np.array(df1['r'] - df1['r'].mean()), fs=240, nperseg=2048)
    # f0 = f[np.argmax(P)]

    
    # fig, ax = plt.subplots()
    # ax.semilogy(f, P)
    # ax.set_xlim(0, 3)
    # ax.set_ylim(1e-3, 10)
    # ax.axvline(f0, ls="--", color="k")
    # # ax.set_xlabel("frequency (Hz)")
    # # ax.set_ylabel("PSD")
    # plt.show()

    # f, t, S = spectrogram(df1['r'] - df1['r'].mean(), fs=240, nperseg=1200, noverlap=1100)
    # plt.pcolormesh(t, f, np.log10(S + 1e-12), shading="auto"); 
    # plt.ylim(0, 3)
    # plt.show()
    # # df2 = calculate_phase(df2)
    
