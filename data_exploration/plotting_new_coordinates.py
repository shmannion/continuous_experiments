from common.paths import DATA, C1, C2, C5, C6
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from scipy.signal import hilbert, welch, spectrogram

"""

The track for the experiment was not aligned with the measurements. As such, the co-ordinates do not vary as we would 
expect. So the first thing we should do is create a new co-ordinate system that makes the variation what we would
expect. We can do this with PCA/SVD.

"""

def calculate_phase(df: pd.DataFrame):
    df.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 
    df['zeta'] = hilbert(df['r'])
    df['zeta_demeaned'] = hilbert(df['r'] - df['r'].mean())
    
    #N.B. This next line is incorrect as of 3/9/26 @ 16:50, unwrap needs to take the band-passed version
    # df['theta'] = np.unwrap(np.angle(df['H']))
    return df

def get_basis_vector(pair, cand):
    paths = sorted(DATA.glob('condition_*/pair_*/p_*_trial_*.csv'))
    dfs = []
    for path in paths:
        m = re.match(rf"p_(\d+)([{cand}])_trial_(\d+)", path.stem)
        if m is None or int(m[1]) != pair or m[2] not in cand:
            continue
        df = pd.read_csv(path, header=None)
        df.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 
        df = df.iloc[240:]
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    xyz = df[["x", "y", "z"]].to_numpy()
    centre = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - centre, full_matrices=False)
    direction = vt[0] # unit vector along the track
    # s = (xyz - centre) @ direction
    return direction, centre




if __name__ == "__main__":
    df = pd.read_csv(f'{C5}/pair_06/p_06A_trial_01.csv', header=None)
    df.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 

    # df1 = calculate_phase(df1)
    v, c = get_basis_vector(6, 'AB') 
    xyz = df[["x", "y", "z"]].to_numpy()
    s = (xyz - c) @ v          # one value per sample, cm along the rail
    t = np.arange(len(s)) / 240   
    
    fig, ax = plt.subplots()
    ax.plot(t, df['r'])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("norm of original co-ordinates")
    ax.set_title("Full trial, original co-ordinates")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(t[240:], df['r'].iloc[240:])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("norm of original co-ordinates")
    ax.set_title("First second discarded, original co-ordinates")
    plt.show()


    # fig, ax = plt.subplots()
    # ax.plot(t, s)
    # ax.set_xlabel("time (s)")
    # ax.set_ylabel("position along rail (cm)")
    # plt.show()

    fig, ax = plt.subplots(2)
    ax[0].plot(t[240:], df['r'].iloc[240:])
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("norm of original co-ordinates")
    ax[1].plot(t, s)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("position along rail (cm)")
    plt.tight_layout()
    plt.show()


    # Now look at two trials in original co-ords, transformed to the same basis, and then to individual bases
    # Original
    df1 = pd.read_csv(f'{C5}/pair_06/p_06B_trial_01.csv', header=None)
    df1.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 
    
    fig, ax = plt.subplots()
    ax.plot(t[240:], df['r'].iloc[240:], label='A')
    ax.plot(t[240:], df1['r'].iloc[240:], label='B')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("norm of original co-ordinates")
    ax.set_title("First second discarded, original co-ordinates")
    plt.show()
    # Transformed to co-ordinates, sharing one basis

    v, c = get_basis_vector(6, 'AB') 
    xyz = df[["x", "y", "z"]].to_numpy()
    s = (xyz - c) @ v      
    t = np.arange(len(s)) / 240   
    
    xyz1 = df1[["x", "y", "z"]].to_numpy()
    s1 = (xyz1 - c) @ v      
    t1 = np.arange(len(s1)) / 240   
    
    fig, ax = plt.subplots()
    ax.plot(t[240:], s[240:], label='A')
    ax.plot(t1[240:], s1[240:], label='B')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance along rail")
    ax.set_title("Shared basis")
    plt.show()

    # Transformed to co-ordinates, not sharing a basis

    vA, cA = get_basis_vector(6, 'A') 
    vB, cB = get_basis_vector(6, 'B') 
    if vA @ vB < 0:
        vB *= -1
    if vA[2] < 0: 
        vA, vB = -vA, -vB
    # if v[2] < 0:
    #     v *= -1
    xyz = df[["x", "y", "z"]].to_numpy()
    s = (xyz - cA) @ vA      
    t = np.arange(len(s)) / 240   
    np.corrcoef(s, df.z)[0, 1]

    
    # vB, cB = get_basis_vector(6, 'B') 
    # if v[2] < 0:
    #     v *= -1
    xyz1 = df1[["x", "y", "z"]].to_numpy()
    s1 = (xyz1 - cB) @ vB     
    t1 = np.arange(len(s1)) / 240   
    
    fig, ax = plt.subplots()
    ax.plot(t[240:], s[240:], label='A')
    ax.plot(t1[240:], s1[240:], label='B')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance along rail")
    ax.set_title("Individual bases")
    plt.show()
    
    # Direct comparison no transformation to transformation to separate bases

    fig, axes = plt.subplots(4)
    axes[0].plot(t[240:], df['r'].iloc[240:], label='A')
    axes[0].plot(t[240:], df1['r'].iloc[240:], label='B')
    axes[1].plot(t[240:], s[240:], label='A')
    axes[1].plot(t1[240:], s1[240:], label='B')
    axes[2].plot(t[240:], s[240:], label='A')
    axes[2].plot(t1[240:], -1 * s1[240:], label='B')
    axes[0].set_xlabel("time (s)")
    axes[1].set_xlabel("time (s)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_ylabel("norm of original co-ordinates")
    axes[1].set_ylabel("transformed co-ordinates")
    axes[2].set_ylabel("transformed co-ordinates, flipped B")
    # ax.set_title("Full trial, original co-ordinates")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2)
    axes[0].plot(df['z'].iloc[240:], s[240:], label='A')
    axes[1].plot(df1['z'].iloc[240:], s1[240:], label='B')
    axes[0].set_xlabel("z")
    axes[1].set_xlabel("z")
    axes[0].set_ylabel("s")
    axes[1].set_ylabel("s")
    plt.tight_layout()
    # ax.set_ylabel("distance along rail")
    # ax.set_title("Individual bases")
    plt.show()




